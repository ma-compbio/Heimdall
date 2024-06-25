



def get_value(dictionary, key, default=False):
    return dictionary.get(key, default)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0): #, dropout: float = 0.1
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.einsum('sbe->bse', pe)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size , seq_len, embedding_dim]``
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, :x.size(1)] # Broadcasting to match input shape
        x = self.dropout(x)
        return x
