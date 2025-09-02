import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

class PointWiseFeedForward(nn.Module):
    """
    A standard two-layer feed-forward network, applied to the last dimension of the input.
    Uses 1D convolutions to mimic linear layers, a common practice in sequence models.
    """
    def __init__(self, hidden_units: int, dropout_rate: float):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FFN.
        Includes a residual connection.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_units).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Conv1D expects (N, C, L), so we transpose the last two dimensions.
        x = inputs.transpose(-1, -2)
        x = self.dropout1(self.conv1(x))
        x = self.relu(x)
        x = self.dropout2(self.conv2(x))
        x = x.transpose(-1, -2)  # Transpose back
        
        # Residual connection
        outputs = x + inputs
        return outputs


class STTransformer(nn.Module):
    """
    A Spatio-Temporal Transformer model for sequential recommendation.
    It encodes a sequence of user-item interactions along with their spatial context.
    """
    def __init__(self, user_num: int, item_num: int, geo_num: int, disnum: int, args):
        super(STTransformer, self).__init__()

        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        self.geo_num = geo_num
        self.dis_num = disnum
        self.dev = args.device

        # Embedding layers
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.geo_emb = nn.Embedding(self.geo_num + 1, args.hidden_units, padding_idx=0)
        self.dis_emb = nn.Embedding(self.dis_num + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # Learnable weight for spatial feature fusion
        self.weight_s = nn.Parameter(torch.randn(args.hidden_units, args.hidden_units))

        # Transformer blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

    def get_embedding_parameters(self) -> list:
        """Returns the weights of the main embedding layers for regularization."""
        return [self.item_emb.weight, self.geo_emb.weight, self.dis_emb.weight]

    def log2feats(self, log_seqs: torch.Tensor, geo_seqs: torch.Tensor, dis_seqs: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input sequences into a sequence of feature vectors.

        Args:
            log_seqs (torch.Tensor): Item interaction sequences.
            geo_seqs (torch.Tensor): Geographical hash sequences.
            dis_seqs (torch.Tensor): Distance sequences.

        Returns:
            torch.Tensor: The encoded feature sequence from the Transformer.
        """
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # Efficiently create position tensor directly on the target device
        positions = torch.arange(log_seqs.shape[1], device=self.dev).expand(log_seqs.shape[0], -1)
        seqs += self.pos_emb(positions)

        # Add spatial features based on config
        if self.args.geo_hash:
            seqs += self.geo_emb(geo_seqs)
        if self.args.distances:
            seqs += self.dis_emb(dis_seqs)
        if self.args.sptia:
            spatial_fusion = self.geo_emb(geo_seqs) + self.dis_emb(dis_seqs)
            seqs += torch.matmul(spatial_fusion, self.weight_s)

        seqs = self.emb_dropout(seqs)

        # Masking for padded items
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # Causal attention mask to prevent attending to future items
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # PyTorch's MHA expects (Seq_len, Batch, Dim)
            seqs_transposed = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs_transposed)
            mha_outputs, _ = self.attention_layers[i](Q, seqs_transposed, seqs_transposed, attn_mask=attention_mask)
            
            # Residual connection and transpose back
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            # Feed-forward block
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def _enrich_item_embedding(self, item_embs: torch.Tensor, geo_pos_seqs: torch.Tensor, dis_pos_seqs: torch.Tensor) -> torch.Tensor:
        """Helper function to add spatial features to item embeddings."""
        if self.args.geo_hash:
            item_embs += self.geo_emb(geo_pos_seqs)
        if self.args.distances:
            item_embs += self.dis_emb(dis_pos_seqs)
        if self.args.sptia:
            spatial_fusion = self.geo_emb(geo_pos_seqs) + self.dis_emb(dis_pos_seqs)
            item_embs += torch.matmul(spatial_fusion, self.weight_s)
        return item_embs

    def forward(self, user_ids: torch.Tensor, log_seqs: torch.Tensor, pos_seqs: torch.Tensor,
                neg_seqs: torch.Tensor, geo_seqs: torch.Tensor, geo_pos_seqs: torch.Tensor,
                dis_seqs: torch.Tensor, dis_pos_seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        Computes logits for positive and negative samples.
        """
        log_feats = self.log2feats(log_seqs, geo_seqs, dis_seqs)

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        
        # Add spatial context to positive and negative item embeddings
        pos_embs = self._enrich_item_embedding(pos_embs, geo_pos_seqs, dis_pos_seqs)
        neg_embs = self._enrich_item_embedding(neg_embs, geo_pos_seqs, dis_pos_seqs) # Assuming same context for pos/neg

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids: torch.Tensor, log_seqs: torch.Tensor, geo_seqs: torch.Tensor,
                dis_seqs: torch.Tensor, item_indices: np.ndarray) -> torch.Tensor:
        """
        Forward pass for inference.
        Computes scores for a list of candidate items.
        """
        log_feats = self.log2feats(log_seqs, geo_seqs, dis_seqs)

        # Use the feature vector of the last item in the sequence for prediction
        final_feat = log_feats[:, -1, :]
        
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        
        # Note: Spatial context of candidate items is not used here.
        # This could be a potential area for future improvement.

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
