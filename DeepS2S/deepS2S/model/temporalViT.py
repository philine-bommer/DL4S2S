import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- (math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalTransformerModel(nn.Module):
    def __init__(self, 
                 input_dim1=1024, 
                 input_dim2=4, 
                 num_classes=4, 
                 hidden_dim=128, 
                 nhead=2, 
                 num_encoder_layers=1, 
                 num_decoder_layers=1, 
                 dropout=0.3):
        super(TemporalTransformerModel, self).__init__()
        
        # Smaller, simpler embeddings
        self.embedding_x1 = nn.Linear(2 * input_dim1, hidden_dim)
        self.embedding_x2 = nn.Linear(input_dim2, hidden_dim)
        
        self.dropout_emb = nn.Dropout(dropout)
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.pos_decoder = PositionalEncoding(hidden_dim, dropout)
        
        # Simpler transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # Process x1
        x1 = x1.view(batch_size, 3, -1)  # [batch_size, 3, 2048]
        x1_emb = self.embedding_x1(x1)   # [batch_size, 3, hidden_dim]
        x1_emb = self.dropout_emb(x1_emb)
        
        # Process x2
        x2_emb = self.embedding_x2(x2)   # [batch_size, 6, hidden_dim]
        x2_emb = self.dropout_emb(x2_emb)
        
        # Encoder input: [seq_len, batch_size, hidden_dim]
        x1_emb = x1_emb.transpose(0,1)
        x1_emb = self.pos_encoder(x1_emb)
        
        # Encoder output
        memory = self.transformer_encoder(x1_emb)
        
        # Decoder input: [seq_len, batch_size, hidden_dim]
        tgt_emb = x2_emb.transpose(0,1)
        tgt_emb = self.pos_decoder(tgt_emb)
        
        output = self.transformer_decoder(tgt_emb, memory)
        
        # Output: [batch_size, 6, num_classes]
        output = self.fc_out(output.transpose(0,1))
        
        return output