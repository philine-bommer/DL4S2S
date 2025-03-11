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
class Index_LSTM(pl.LightningModule):

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
        super(Index_LSTM, self).__init__()

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
            decoder_input_dim = in_time_lag * out_size


        self.decoder_input_dim =decoder_input_dim
        self.decoder = nn.Sequential(
                nn.LSTM(input_size=decoder_input_dim, hidden_size=decoder_hidden_dim, batch_first=True),
                TimeDistributed(nn.Linear(decoder_hidden_dim, self.out_size), batch_first=True)
            )

        self.MCE = MulticlassCalibrationError(out_size, n_bins=int(self.bs*0.2), norm='l1')
                
    
    def forward(self, x_2d, x_1d):

        x = x_1d[:,:,:self.out_size -1]

        if self.norm: 
            # Min-Max norm  (x-x.min())/(x.max()-x.min()) 
            # -> i.e. range [0,1] for both
            x = (x-x.min())/(x.max()-x.min())
        
        x = x_1d.reshape(x_1d.shape[0], -1).unsqueeze(1).repeat(1, self.out_time_lag, 1)

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
        acc, _, ece = self.get_accuracy(logits, labels)
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
        inputs, _= batch
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




