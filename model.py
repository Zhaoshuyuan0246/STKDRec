import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, geo_num, disnum, args):
        super(SASRec, self).__init__()

        self.args = args

        self.user_num = user_num
        self.item_num = item_num
        self.geo_num = geo_num
        self.dis_num = disnum
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.geo_emb = torch.nn.Embedding(self.geo_num+1, args.hidden_units, padding_idx=0)
        self.dis_emb = torch.nn.Embedding(self.dis_num+1, args.hidden_units, padding_idx=0)

        self.weight_s = nn.Parameter(torch.randn(args.hidden_units, args.hidden_units))

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        if args.fus == 'cat':
            self.fus_linear = torch.nn.Linear(args.hidden_units * 2, args.hidden_units) 

    def get_embedding_parameters(self):
        return [self.item_emb.weight, self.geo_emb.weight, self.dis_emb.weight, self.type_emb.weight, self.t_dif_emb.weight]


    def log2feats(self, log_seqs, geo_seqs, dis_seqs):

        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))

        # Add geo
        if self.args.geo_hash == True:
            seqs += self.geo_emb(geo_seqs)
        # Add dis
        if self.args.distances == True:
            seqs += self.dis_emb(dis_seqs)        # Add sptia
        if self.args.sptia == True:
            seqs += torch.matmul((self.geo_emb(geo_seqs) + self.dis_emb(dis_seqs)), self.weight_s)

        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, geo_seqs, geo_pos_seqs, dis_seqs, dis_pos_seqs): # for training        
        
        log_feats = self.log2feats(log_seqs, geo_seqs, dis_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        
        # Add Geo
        if self.args.geo_hash == True:
            geo_pos_embs = self.geo_emb(geo_pos_seqs)
            pos_embs += geo_pos_embs
            neg_embs += geo_pos_embs
        # Add dis
        if self.args.distances == True:
            dis_pos_embs = self.dis_emb(dis_pos_seqs)
            pos_embs += dis_pos_embs
            neg_embs += dis_pos_embs
        # Add sptia
        if self.args.sptia == True:
            sptia_pos_embs = torch.matmul((self.dis_emb(dis_pos_seqs) + self.geo_emb(geo_pos_seqs)), self.weight_s)
            pos_embs += sptia_pos_embs
            neg_embs += sptia_pos_embs

        if self.args.fus == 'kd' or self.args.fus == 'None':
            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
            return pos_logits, neg_logits # pos_pred, neg_pred
        else:
            return log_feats, pos_embs, neg_embs

    def predict(self, user_ids, log_seqs, geo_seqs, dis_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs, geo_seqs, dis_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        if self.args.fus == 'kd' or self.args.fus == 'None':
            # 加上空间信息
        
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

            return logits # preds # (U, I)
        else:
            return final_feat, item_embs
