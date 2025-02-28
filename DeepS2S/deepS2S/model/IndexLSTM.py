import utils
from typing import Any
import pdb


import numpy as np
import torch
from torch import nn
import lightning as pl
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.classification import MulticlassCalibrationError



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################
##### MODEL WRAPPER #####
#############################
class TimeDistributed(pl.LightningModule):
    def __init__(self, module, batch_first=False, non_linear = False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.non_linear = non_linear

    def forward(self, x):
        # lstm return tuple
        # use first member of output only, gives all hidden states
        if isinstance(x, tuple):
            x = x[0]
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)
        if self.non_linear:
            y = nn.ReLU()(y)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

    
class LSTM_decoder(pl.LightningModule):

    def __init__(self, 
                 LSTM_input_size, 
                 hidden_size, 
                 learning_rate, 
                 weight_decay, 
                 optimizer, 
                 criterion, 
                 out_time_lag,
                 cls_wgt,
                 out_size=4, 
                 clbrt = False,
                 bs = 32,
                 gamma = {},
                 output_probabilities=True):
        
        super(LSTM_decoder, self).__init__()

        self.save_hyperparameters()
        if output_probabilities:
            self.output_activation = nn.Softmax(dim=2)
        else:
            self.output_activation = None

        self.out_time_lag = out_time_lag
        self.lr = learning_rate 
        self.weight_decay = weight_decay
        self.class_weight = cls_wgt
        self.optm = optimizer
        self.crit = criterion
        self.out_size = out_size
        self.clbrt = clbrt
        self.gamma = gamma
        self.bs = bs

        self.decoder = nn.Sequential(
            nn.LSTM(LSTM_input_size, hidden_size=hidden_size, batch_first=True),
            TimeDistributed(nn.Linear(hidden_size, self.out_size), batch_first=True)
        )

        self.MCE = MulticlassCalibrationError(out_size, n_bins=int(self.bs*0.2), norm='l1')
        

    def forward(self, _, x):
        if x.ndim == 3:
            x = torch.argmax(x, axis=-1).to(dtype=torch.float32)
        x = x.unsqueeze(1).repeat(1, self.out_time_lag, 1)
        x = self.decoder(x)
        x = torch.squeeze(x)
        if self.output_activation:
            x = self.output_activation(x)
        
        return x
    
    def configure_optimizers(self):
        optimizer =  self.optm(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.optimizer = optimizer
        return optimizer 
    
    def configure_loss(self):
        """_summary_

        Args:
            clbrt (bool, optional): enable/ disable calibration using Focal Loss 
            with adaptive gamma. Defaults to False.
            Reference:
            [1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
                arXiv preprint arXiv:1708.02002, 2017.

        Returns:
            _type_: loss criterion
        """
        if self.clbrt:
            criterion = self.crit
        else:
            criterion = self.crit(weight=torch.Tensor(self.class_weight))
        self.criterion = criterion
        return criterion 

    def training_step(self,
          batch,
          batch_idx):

        inputs, labels = batch #utils.move_tuple_to(inputs, device), labels.to(device)
        logits = self.forward(inputs[0],inputs[1])
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        acc, certainty, ece = self.get_accuracy(logits, labels)
        values = {'train_loss': loss, 'train_ece': ece,'train_acc': acc}
        self.log_dict(values, sync_dist=True)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        inputs,y = val_batch
        logits = self.forward(inputs[0],inputs[1])
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        acc, certainty, ece = self.get_accuracy(logits, y)
        values = {'val_loss': loss, 'val_ece': ece, 'val_acc': acc, 'val_certainty': certainty}  # add more items if needed
        self.log_dict(values, sync_dist=True)
    
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):

        inputs,y = batch #utils.move_tuple_to(inputs, device), labels.to(device)
        logits = self.forward(inputs[0],inputs[1])
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        acc, certainty, ece = self.get_accuracy(logits, y)
        values = {"test_ece": ece, "test_acc": acc, "test_certainty": certainty}  # add more items if needed
        self.log_dict(values, sync_dist=True)

        return {"test_acc": acc, "test_certainty": certainty}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
                    inputs, labels = batch
                    logits = self.forward(inputs[0],inputs[1])
                    return logits
    
    def get_accuracy(self, logits, y):
        predictions = logits.detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        certainty = np.mean(np.max(predictions, axis=-1))
        class_predictions = np.argmax(predictions, axis=-1)
        accuracy = balanced_accuracy_score(labels.flatten().astype(int), class_predictions.flatten().astype(int))
        ece = self.MCE(logits.detach().cpu().reshape(-1, predictions.shape[-1]), y.detach().cpu().flatten()) 

        return float(accuracy), float(certainty), float(ece)




#############################
##### MULTIMODAL ViT-LSTM #####
#############################
class spatiotemporal_Neural_Network(pl.LightningModule):

    def __init__(self,
                 encoder_u,
                 encoder_sst,
                 enc_out_shape,
                 in_time_lag,
                 out_time_lag,
                 decoder_hidden_dim,
                 learning_rate,
                 optim,
                 weight_decay,
                 cls_wgt,
                 criterion,
                 out_dim=1,
                 dropout=0.0,
                 output_probabilities=False,
                 norm_both = False,
                 norm = True,
                 norm_bch = False,
                 add_attn = False,
                 n_heads = 3,
                 clbrt = False,
                 bs = 32,
                 gamma = {},
                 out_size = 4,
                 mode = 'run',
                 scale = False,
                 factor = 1,
                 ts_len = 4,

            ):
        """ Generates a model that takes a multimodal time series as input and outputs a time series.
            The elements of output time series may either be a contiuous variable or probabilities 
            for a certain number of classes.

        Args:
            input_dim (int): Number of channels of 2D input TS.
            hidden_dim (int): Size of hidden dimension of ConvLSTM.
            encoder_num_layers (int): Number of ConvLSTM layers.
            in_time_lag (int): Length of input time series.
            out_time_lag (int): Lentgh of output time series.
            decoder_hidden_dim (int): Size of hidden dimension of decoder LSTM.
            kernel_size (int): Size of kernel of ConvLSTM.
            frame_size (tuple): Size of 2D input TS.
            out_dim (int, optional): Number of classes in case of classification task, else 1. Defaults to 1.
            maxpool_kernel_size (int, optional): Size of Maxpool kernel. Defaults to None.
            dropout (float, optional): Fraction of activations set to zero after every layer. Defaults to 0.0.
            output_probabilities (bool, optional): If True, softmax activation is applied to final output to produce 
                                                   probabilities for each class and time step. Defaults to False.
            mode (str): regimes - random regimes, encoder - random encoder,  base - LSTM baseline, run (default) - STNN
            scale (bool, optional): True - add linear layer to upsample regimes
        """
        super(spatiotemporal_Neural_Network, self).__init__()

        self.save_hyperparameters(ignore=['encoder_sst','encoder_u'])

        self.norm_both = norm_both
        self.norm_bch = norm_bch
        self.out_time_lag = out_time_lag
        self.learning_rate = learning_rate
        self.optm = optim
        self.weight_decay = weight_decay
        self.class_weight = cls_wgt
        self.crit = criterion
        self.norm = norm
        self.out_size = out_size
        self.add_attn = add_attn
        self.attn_heads = n_heads
        self.clbrt = clbrt
        self.bs = bs
        self.gamma = gamma
        self.mode = mode
        self.scale = scale
        self.factor = factor
        if 'encoder' in mode:
            print('Test encoder impact with random variables')
        elif 'regimes' in mode:
            print('Test regime impact with random variables')
        if scale:
            print('Upsample regimes!')
        
        if output_probabilities:
            self.output_activation = nn.Softmax(dim=2)
        else:
            self.output_activation = None

        self.encoder_u = encoder_u
        self.encoder_sst = encoder_sst
        encoder_out_dim = enc_out_shape
        self.ts_len = ts_len

        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(dropout)

        
        if add_attn:
            if self.encoder_u is None:
                self.multihead = nn.MultiheadAttention(self.out_time_lag, self.attn_heads, batch_first=True)

                decoder_input_dim = encoder_out_dim[0] * encoder_out_dim[1] + self.ts_len
            else:
                self.multihead = nn.MultiheadAttention(self.out_time_lag, self.attn_heads, batch_first=True)

                decoder_input_dim = encoder_out_dim[0] * 2 * encoder_out_dim[1] + self.ts_len

        else: 
            if 'base' in self.mode:
                decoder_input_dim = in_time_lag * out_size

            elif self.encoder_u is None or self.encoder_sst is None:
                if self.encoder_u is None and self.encoder_sst is None:
                    decoder_input_dim = self.ts_len
                elif 'images' in self.mode:
                    decoder_input_dim = utils.prod(encoder_out_dim)
                else:
                    decoder_input_dim = utils.prod(encoder_out_dim) + self.ts_len
                
                if self.scale:
                    self.upsample = nn.Linear(in_time_lag * out_dim + 1, 2*utils.prod(encoder_out_dim))
                    decoder_input_dim = 2*2*utils.prod(encoder_out_dim)

            else:
                if 'images' in self.mode:
                    decoder_input_dim = 2*utils.prod(encoder_out_dim)
                else:
                    decoder_input_dim = 2*utils.prod(encoder_out_dim) + out_dim
                
                if self.scale:
                    self.upsample = nn.Linear(in_time_lag * out_dim, 2*utils.prod(encoder_out_dim))
                    decoder_input_dim = 2*2*utils.prod(encoder_out_dim)

        self.decoder_input_dim =decoder_input_dim
        if 'linear' in self.mode:
            self.decoder = nn.Sequential(
            TimeDistributed(nn.Linear(decoder_input_dim, self.out_size), batch_first=True, non_linear=False),
            )
        else:
            self.decoder = nn.Sequential(
                nn.LSTM(input_size=decoder_input_dim, hidden_size=decoder_hidden_dim, batch_first=True),
                TimeDistributed(nn.Linear(decoder_hidden_dim, self.out_size), batch_first=True)
            )

        self.MCE = MulticlassCalibrationError(out_size, n_bins=int(self.bs*0.2), norm='l1')
                
    
    def forward(self, x_2d, x_1d):

        if not x_2d:
            x = x_1d.reshape(x_1d.shape[0], self.out_time_lag , self.ts_len)

        elif x_2d.shape[2]>1:

            x = self.encode_both(x_2d)
            x_1d = x_1d.reshape(x_1d.shape[0], self.out_time_lag , self.out_size)

        elif not self.encoder_u:

            x_sst = x_2d
            x = self.encode(x_sst, self.encoder_sst)
            x_1d = x_1d.reshape(x_1d.shape[0], self.out_time_lag , self.ts_len)


        elif not self.encoder_sst:

            x_u = x_2d
            x = self.encode(x_u, self.encoder_u)
            x_1d = x_1d.reshape(x_1d.shape[0], self.out_time_lag , self.ts_len)

            if 'base' in self.mode:
                x = x_1d[:,:,:self.out_size -1]




        if self.norm: 
            # Min-Max norm  (x-x.min())/(x.max()-x.min()) 
            # -> i.e. range [0,1] for both
            x = (x-x.min())/(x.max()-x.min())
        if self.add_attn:
             ##Attention
             X = torch.cat((x, x_1d), dim=2)
             q_x = np.swapaxes(X, 1,2)#X
             v_k = np.swapaxes(x_1d,1,2) #values are weighted according to Softmax(QK)
             x_enc, attn_weights = self.multihead(q_x,  v_k,  v_k)
             self.attention_wghts = attn_weights
             x_enc = np.swapaxes(x_enc,1,2)

        else:
            if 'base' in self.mode:
                x = x_1d.reshape(x_1d.shape[0], -1).unsqueeze(1).repeat(1, self.out_time_lag, 1)
            elif 'images' in self.mode:
                x = x
            else:
                x = torch.cat((x, x_1d), dim=2)

                if self.norm_both: 
                    # Standardize joint vectors: (x - x.mean())/x.std()
                    x = (x - x.mean())/x.std() 

            if not x_2d:
                x_enc = x_1d.reshape(x_1d.shape[0], self.out_time_lag , self.ts_len)
            elif 'base' in self.mode:
                x_enc = x
            else: 
                x_enc = x

        x= self.decoder(x_enc)
        x = torch.squeeze(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

    def configure_optimizers(self):

        optimizer = self.optm(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.optimizer = optimizer
        return optimizer 
    
    def configure_loss(self):
        """_summary_

        Args:
            clbrt (bool, optional): enable/ disable calibration using Focal Loss 
            with adaptive gamma. Defaults to False.
            Reference:
            [1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
                arXiv preprint arXiv:1708.02002, 2017.

        Returns:
            _type_: loss criterion
        """
        if self.clbrt:
            criterion = self.crit
        else:
            criterion = self.crit(weight=torch.Tensor(self.class_weight))
        self.criterion = criterion
        return criterion 


    def training_step(self,
          batch,
          batch_idx):
        """ Training a given model on the provided training data.

        Args:
            model (torch.nn.Module): Model architecture to train
            training_data_loader (torch.DatasetLoader): A dataset loader that contains all training data.
            criterion (torch.nn.loss): Loss funtion
            optimizer (torch.optim): Optimizer
            epochs (int): Number of epochs. If None run until model does not improve for 3 epochs.
            print_freq (int): Number of batches after which a print is made.
        """
        inputs, labels = batch
        logits = self.forward(inputs[0],inputs[1])
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        acc, certainty, ece = self.get_accuracy(logits, labels)
        values = {'train_loss': loss, 'train_ece': ece,'train_acc': acc}
        self.log_dict(values, sync_dist=True)


        return loss
    
    
    def validation_step(self, batch, batch_idx):

        inputs, labels = batch 
        logits = self.forward(inputs[0],inputs[1])
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        acc, certainty, ece = self.get_accuracy(logits, labels)
        values = {"val_loss": loss,"val_ece": ece, "val_acc": acc, "val_certainty": certainty}  # add more items if needed
        self.log_dict(values, sync_dist=True)
    
        return {"val_loss": loss, "val_accuracy": acc}
    
    def test_step(self, batch, batch_idx):

        inputs, labels = batch 
        logits = self.forward(inputs[0],inputs[1])
        acc, certainty, ece = self.get_accuracy(logits, labels)
        values = {"test_ece": ece, "test_acc": acc, "test_certainty": certainty}  # add more items if needed
        self.log_dict(values, sync_dist=True)

        return {"test_acc": acc, "test_certainty": certainty, "test_ece": ece}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Perform a forward pass on the model to generate predictions.

        Args:
            batch: A tuple containing the inputs and labels.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            The predicted logits.
        """
        inputs, labels = batch
        logits = self.forward(inputs[0],inputs[1])
        return logits
    
    def get_accuracy(self, logits, y):
        predictions = logits.detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        certainty = np.mean(np.max(predictions, axis=-1))
        class_predictions = np.argmax(predictions, axis=-1)
        accuracy = balanced_accuracy_score(labels.flatten().astype(int), class_predictions.flatten().astype(int))
        ece = self.MCE(logits.detach().cpu().reshape(-1, predictions.shape[-1]), y.detach().cpu().flatten()) 
         
        return float(accuracy), float(certainty), float(ece)
    
    def freeze_encoder(self):
        if self.encoder_u is None:
            self.encoder_sst.freeze()
        elif self.encoder_sst is None:
            self.encoder_u.freeze()
        else:
            self.encoder_u.freeze()
            self.encoder_sst.freeze()

        return "Encoder frozen"
    
    def encode(self, x, encoder):
    
        for t in range(x.shape[1]):
            if t == 0:
                x_enc = encoder.get_image_embedding(x[:,t,...])
                x_enc = x_enc.reshape(x_enc.shape[0], -1)[:,None,...]
            else:
                enc = encoder.get_image_embedding(x[:,t,...]).reshape(x_enc.shape[0], -1)[:,None,...]
                x_enc = torch.cat((x_enc, enc), dim=1)

        return x_enc
    
    def encode_both(self, x_2d):

        x_sst = x_2d[:,:,0,:,:]
        x_u = x_2d[:,:,1,:,:]

        for t in range(x_2d.shape[1]):
            if t == 0:
                x_enc_s = self.encoder_sst.get_image_embedding(x_sst[:,None,t,...])
                x_enc_s = x_enc_s.reshape(x_enc_s.shape[0], -1)[:,None,...]

                x_enc_u = self.encoder_u.get_image_embedding(x_u[:,None,t,...])
                x_enc_u = x_enc_u.reshape(x_enc_u.shape[0], -1)[:,None,...]
            else:
                enc_s = self.encoder_sst.get_image_embedding(x_sst[:,None,t,...]).reshape(x_enc_s.shape[0], -1)[:,None,...]
                x_enc_s = torch.cat((x_enc_s, enc_s), dim=1)

                enc_u = self.encoder_u.get_image_embedding(x_u[:,None,t,...]).reshape(x_enc_u.shape[0], -1)[:,None,...]
                x_enc_u = torch.cat((x_enc_u, enc_u), dim=1)
        
        x = torch.cat((x_enc_s, x_enc_u), dim=2)

        return x


