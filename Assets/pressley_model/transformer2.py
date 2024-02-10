# I forgot where I got this from. Probably pure GPT4
# This has encoder so thats why not using it
import torch
import torch.nn as nn
import math

# Define positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Define transformer model
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        self.d_model = 512 # Embedding dimension
        self.nhead = 8 # Number of attention heads
        self.num_encoder_layers = 6 # Number of encoder layers
        self.num_decoder_layers = 6 # Number of decoder layers
        self.dim_feedforward = 2048 # Dimension of feedforward network
        self.dropout = 0.1 # Dropout probability
        self.max_len = 100 # Maximum length of input sequence
        self.src_vocab_size = 256
        self.tgt_vocab_size = 256

        # Define encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        decoder_layer = nn.TransformerDecoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)

        # Define encoder and decoder
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)

        # Define positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.max_len)
        self.pos_decoder = PositionalEncoding(self.d_model, self.dropout, self.max_len)

        # Define input and output embeddings
        self.src_embed = nn.Embedding(self.src_vocab_size, self.d_model)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, self.d_model)

        # Define output linear layer
        self.linear = nn.Linear(self.d_model, self.tgt_vocab_size)

        # Initialize parameters
        self.init_weights()

    def init_weights(self):
        # Initialize embeddings with normal distribution
        initrange = 0.1
        self.src_embed.weight.data.uniform_(-initrange, initrange)
        self.tgt_embed.weight.data.uniform_(-initrange, initrange)

        # Initialize linear layer with zeros
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed and encode the source sequence
        src = self.src_embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # Embed and decode the target sequence
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask,
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        # Apply linear layer to get logits
        output = self.linear(output)

        return output
    