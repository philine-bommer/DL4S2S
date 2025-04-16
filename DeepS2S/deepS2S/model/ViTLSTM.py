from typing import Any
import pdb


import numpy as np

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import lightning as pl
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.classification import MulticlassCalibrationError
from ..utils.utils import prod

" Functions from .py files"
# from ViT import pair

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        # Apply temperature scaling
        return logits / self.temperature

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


#############################
##### MULTIMODAL ViT-LSTM #####
#############################
class Encoder_ViT(nn.Module):
    def __init__(self, encoder_sst, encoder_u, **params):
        """
        Initializes the Encoder_ViT class.
        Args:
            encoder_sst: The encoder for olr.
            encoder_u: The encoder for u.
            **params: Additional parameters.
        Returns:
            None
        """
        
        super(Encoder_ViT, self).__init__()

        self.encoder_sst = encoder_sst
        self.encoder_u = encoder_u
        self.params = params

        if params.get('vit', False):
            self.params = params['vit']
            self.create_patches()
        else:
            self.params = [] 


    def forward(self, x_2d):
        """
        Encodes the input images using the encoder networks for sea surface temperature (olr) and u-component of wind (u).
        Args:
            x_2d (Tensor): Input images of shape (batch_size, sequence_length, channels, height, width).
        Returns:
            Tuple[Tensor, Tensor]: Encoded representations of olr and u-component of wind, respectively.
                The shape of each output tensor is (batch_size, sequence_length, encoded_dim).
        """

        x_olr = x_2d[:,:,0,:,:]
        x_u = x_2d[:,:,1,:,:]

        
        for t in range(x_2d.shape[1]):
            if t == 0:
                x_enc_s = self.encoder_sst.encode(x_olr[:,None,t,...])
                x_enc_s = x_enc_s.reshape(x_enc_s.shape[0], -1)[:,None,...]

                x_enc_u = self.encoder_u.encode(x_u[:,None,t,...])
                x_enc_u = x_enc_u.reshape(x_enc_u.shape[0], -1)[:,None,...]
            else:
                enc_s = self.encoder_sst.encode(x_olr[:,None,t,...]).reshape(x_enc_s.shape[0], -1)[:,None,...]
                x_enc_s = torch.cat((x_enc_s, enc_s), dim=1)

                enc_u = self.encoder_u.encode(x_u[:,None,t,...]).reshape(x_enc_u.shape[0], -1)[:,None,...]
                x_enc_u = torch.cat((x_enc_u, enc_u), dim=1)

        return x_enc_s, x_enc_u
    
    def create_patches(self):

        image_height, image_width = pair(self.frame_size)
        patch_height, patch_width = pair(tuple(self.params['patch_size']))
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = self.params['channels'] * patch_height * patch_width
        assert self.params.get('pool','cls') in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.params['dim']),
            nn.LayerNorm(self.params['dim']),
        )

        self.pool = self.params.get('pool','cls')
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(self.params['dim'], self.params['num_classes'])

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.params['dim']))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.params['dim']))


class ViT_LSTM(pl.LightningModule):

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
                 clbrt = False,
                 bs = 32,
                 gamma = {},
                 out_size = 4,
                 mode = 'run',
                 frame_size = (22, 256),
                 **params
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
            factor (int, optional): factor to upsample regimes
            **params: Additional parameters for the model.
        """
        super(ViT_LSTM, self).__init__()

        self.save_hyperparameters(ignore=['encoder_u','encoder_sst'])

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
        self.clbrt = clbrt
        self.bs = bs
        self.gamma = gamma
        self.mode = mode
        self.frame_size = frame_size

        
        if output_probabilities:
            self.output_activation = nn.Softmax(dim=2)
        else:
            self.output_activation = None

        self.encoder_u = encoder_u
        self.encoder_sst = encoder_sst
        encoder_out_dim = enc_out_shape

        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(dropout)
        self.norm_layer = nn.BatchNorm1d(in_time_lag)

        if params.get('vit', False):
            self.params = params['vit']
            self.create_patches()
        else:
            self.params = [] 
        

        if 'base' in self.mode:
            decoder_input_dim = in_time_lag * out_dim
        else:
            decoder_input_dim = 2*prod(encoder_out_dim) + out_dim
        
        
        self.decoder_input_dim =decoder_input_dim
        self.decoder = nn.Sequential(
                nn.LSTM(input_size=decoder_input_dim, hidden_size=decoder_hidden_dim, batch_first=True),
                TimeDistributed(nn.Linear(decoder_hidden_dim, self.out_size), batch_first=True)
            )
     
        self.MCE = MulticlassCalibrationError(out_size, n_bins=int(self.bs*0.2), norm='l1')
    
    def create_patches(self):

        image_height, image_width = pair(self.frame_size)
        patch_height, patch_width = pair(tuple(self.params['patch_size']))
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = self.params['channels'] * patch_height * patch_width
        assert self.params.get('pool','cls') in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.params['dim']),
            nn.LayerNorm(self.params['dim']),
        )

        self.pool = self.params.get('pool','cls')
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(self.params['dim'], self.params['num_classes'])

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.params['dim']))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.params['dim']))
                
    
    def forward(self, x_2d, x_1d):

        x_enc_s, x_enc_u = self.encode_images(x_2d)
   
        x = torch.cat((x_enc_s, x_enc_u), dim=2)
        
        if self.norm: 
            # Min-Max norm  (x-x.min())/(x.max()-x.min()) 
            # -> i.e. range [0,1] for both
            x = (x-x.min())/(x.max()-x.min())

        if 'base' in self.mode:
            x = x_1d.reshape(x_1d.shape[0], -1).unsqueeze(1).repeat(1, self.out_time_lag, 1)
        else:
            x = torch.cat((x, x_1d), dim=2)

            if self.norm_both: 
                # Standardize joint vectors: (x - x.mean())/x.std()
                x = (x - x.mean())/x.std() 
        
        x_enc = x
        self.encoded_input_data = x_enc
        if not self.encoded_input_data.requires_grad:
            self.encoded_input_data.requires_grad = True
            self.encoded_input_data.retain_grad()
        x_enc = self.norm_layer(x_enc)
        x_enc = self.dropout(x_enc) 
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
        """
        Performs a validation step on a batch of data.
        Args:
            batch: A tuple containing the inputs and labels for the batch.
            batch_idx: The index of the current batch.
        Returns:
            A dictionary containing the validation loss and accuracy.
        """
        inputs, labels = batch 
        logits = self.forward(inputs[0],inputs[1])
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        acc, certainty, ece = self.get_accuracy(logits, labels)
        values = {"val_loss": loss,"val_ece": ece, "val_acc": acc, "val_certainty": certainty}  # add more items if needed
        self.log_dict(values, sync_dist=True)
    
        return {"val_loss": loss, "val_accuracy": acc}
    
    def test_step(self, batch, batch_idx):
        """
        Perform a single testing step.

        Args:
            batch: A tuple containing the inputs and labels.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the test accuracy, test certainty, and test expected calibration error (ECE).
        """
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
        """
        Calculates the accuracy, certainty, and expected calibration error (ECE) of the model's predictions.
        Parameters:
            logits (Tensor): The predicted logits from the model.
            y (Tensor): The ground truth labels.
        Returns:
            accuracy (float): The balanced accuracy score of the predictions.
            certainty (float): The average certainty of the predictions.
            ece (float): The expected calibration error of the predictions.
        """

        predictions = logits.detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        certainty = np.mean(np.max(predictions, axis=-1))
        class_predictions = np.argmax(predictions, axis=-1)
        accuracy = balanced_accuracy_score(labels.flatten().astype(int), class_predictions.flatten().astype(int))
        ece = self.MCE(logits.detach().cpu().reshape(-1, predictions.shape[-1]), y.detach().cpu().flatten()) 
         
        return float(accuracy), float(certainty), float(ece)
    
    def freeze_encoder(self):
        """
        Freezes the encoder by freezing the encoder_u and encoder_sst.

        Returns:
            str: A message indicating that the encoder has been frozen.
        """

        self.encoder_u.freeze()
        self.encoder_sst.freeze()

        return "Encoder frozen"
    
    def encode_images(self, x_2d):
        """
        Encodes the input images using the encoder networks for sea surface temperature (olr) and u-component of wind (u).
        Args:
            x_2d (Tensor): Input images of shape (batch_size, sequence_length, channels, height, width).
        Returns:
            Tuple[Tensor, Tensor]: Encoded representations of olr and u-component of wind, respectively.
                The shape of each output tensor is (batch_size, sequence_length, encoded_dim).
        """

        x_olr = x_2d[:,:,0,:,:]
        x_u = x_2d[:,:,1,:,:]

        
        for t in range(x_2d.shape[1]):
            if t == 0:
                x_enc_s = self.encoder_sst.encode(x_olr[:,None,t,...])
                x_enc_s = x_enc_s.reshape(x_enc_s.shape[0], -1)[:,None,...]

                x_enc_u = self.encoder_u.encode(x_u[:,None,t,...])
                x_enc_u = x_enc_u.reshape(x_enc_u.shape[0], -1)[:,None,...]
            else:
                enc_s = self.encoder_sst.encode(x_olr[:,None,t,...]).reshape(x_enc_s.shape[0], -1)[:,None,...]
                x_enc_s = torch.cat((x_enc_s, enc_s), dim=1)

                enc_u = self.encoder_u.encode(x_u[:,None,t,...]).reshape(x_enc_u.shape[0], -1)[:,None,...]
                x_enc_u = torch.cat((x_enc_u, enc_u), dim=1)

        return x_enc_s, x_enc_u
    
   