# A Decoder only transformer 
# converted from https://github.com/google-research/robotics_transformer/blob/master/transformer.py to pytorch by gpt4
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List

class _TransformerLayer(nn.Module):
    """A single transformer block."""

    def __init__(self,
                 layer_size: int = 512,
                 num_heads: int = 8,
                 feed_forward_size: int = 512,
                 dropout_rate: float = 0.1,
                 return_attention_scores: bool = False):
        """Creates a Transformer layer.
        """
        super(_TransformerLayer, self).__init__()

        self.layernorm1 = nn.LayerNorm(layer_size, eps=1e-6)
        self.mha1 = nn.MultiheadAttention(layer_size, num_heads, dropout_rate)
        self.ff = nn.Linear(layer_size, feed_forward_size)
        self.layernorm2 = nn.LayerNorm(feed_forward_size, eps=1e-6)
        self.dropout_ff = nn.Dropout(dropout_rate)
        self._return_attention_scores = return_attention_scores

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor,
                training: bool) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Calls the layer.

        Args:
          x: Input Tensor of shape `(B, T, dim)`.

        Returns:
          y: Output Tensor of shape `(B, T, dim)`. Also return the attention scores
          of shape `(B, T, dim)` or None.
        """
        x1 = self.layernorm1(x)
        mha_results = self.mha1(
            query= x1,
            key= x1,
            value= x1,
            attn_mask=attention_mask,
            need_weights=self._return_attention_scores)
        if self._return_attention_scores:
            x1, score = mha_results
        else:
            x1, score = mha_results[0], None

        x = x + x1

        y = self.layernorm2(x)
        ff_y = self.ff(y)
        ff_y = self.dropout_ff(ff_y)
        x = x + ff_y
        return x, score

class Transformer(nn.Module):
    """A decoder only transformer."""

    def __init__(self,
                 num_layers: int = 1,
                 layer_size: int = 512,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 vocab_size: int = 256,
                 context_length: int = 5,
                 token_embedding_size: int = 512,
                 return_attention_scores: bool = False):
        """Creates a transformer.
        """
        super(Transformer, self).__init__()

        self._layers = nn.ModuleList([
            _TransformerLayer(  # pylint: disable=g-complex-comprehension
                layer_size=layer_size,
                num_heads=num_heads,
                feed_forward_size=token_embedding_size,
                dropout_rate=dropout_rate,
                return_attention_scores=return_attention_scores)
            for _ in range(num_layers)
        ])

        self._token_emb = nn.Linear(token_embedding_size, token_embedding_size)
        self._position_emb = nn.Linear(context_length, token_embedding_size)
        self._output_tokens = nn.Linear(token_embedding_size, vocab_size)


    def forward(
        self,
        x: torch.Tensor,
        training: bool,
        attention_mask: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Calls the layer.
        """

        seq_len = x.size(1)

        x = self._token_emb(x)
        x += self._position_emb(torch.arange(seq_len, device='cuda', dtype=torch.float))
        scores = []

        for layer in self._layers:
            x, score = layer(x, attention_mask=attention_mask, training=training)
            if score is not None:
                scores.append(score)
        x = self._output_tokens(x)
        return x, scores