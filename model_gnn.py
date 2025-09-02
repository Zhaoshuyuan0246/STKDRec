# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, constant_
from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch.nn import Module
import torch.nn.functional as F

class GlobalGNN(Module):
    def __init__(self, args):
        super(GlobalGNN, self).__init__()
        self.args = args
        self.gnn_hidden_units = args.gnn_hidden_units
        in_channels = hidden_channels = self.gnn_hidden_units
        self.num_layers = len(args.sample_size)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.gcn = GCNConv(self.gnn_hidden_units, self.gnn_hidden_units)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))

    def forward(self, x, adjs, attr, sample_type='sparse'):
        xs = []
        x_all = x
        if self.num_layers > 1:
            for i, (edge_index, e_id, size) in enumerate(adjs):
                weight = attr[e_id].view(-1).type(torch.float)

                x = x_all
                if len(list(x.shape)) < 2:
                    x = x.unsqueeze(0)
                
                if sample_type == 'sparse':
                    edge_index_list = edge_index.coo()
                    edge_index = torch.stack((edge_index_list[1], edge_index_list[0]))
                    x = self.gcn(x, edge_index, weight)
                else:
                    x = self.gcn(x, edge_index, weight)

                # sage
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
        else:
            # 只有1-hop的情況
            edge_index, e_id, size = adjs.edge_index, adjs.e_id, adjs.size
            x = x_all
            x = self.dropout(x)
            weight = attr[e_id].view(-1).type(torch.float)
            if len(list(x.shape)) < 2:
                x = x.unsqueeze(0)
            x = self.gcn(x, edge_index, weight)
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[-1]((x, x_target), edge_index)
        xs.append(x)
        return torch.cat(xs, 0)


class STKGEncoder(nn.Module):
    def __init__(self, args, global_graph):
        super(STKGEncoder, self).__init__()
        self.args = args
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device(args.device)
        self.global_graph = global_graph.to(self.device)
        self.global_gnn = GlobalGNN(args)

        self.user_embeddings = nn.Embedding(args.user_size+1, args.gnn_hidden_units)
        self.item_embeddings = nn.Embedding(args.item_size+1, args.gnn_hidden_units, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.maxlen, args.gnn_hidden_units)

        # AttNet
        self.w_1 = nn.Parameter(torch.Tensor(2*args.gnn_hidden_units, args.gnn_hidden_units))
        self.w_2 = nn.Parameter(torch.Tensor(args.gnn_hidden_units, 1))
        self.linear_1 = nn.Linear(args.gnn_hidden_units, args.gnn_hidden_units)
        self.linear_2 = nn.Linear(args.gnn_hidden_units, args.gnn_hidden_units, bias=False)

        self.LayerNorm = nn.LayerNorm(args.gnn_hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear_transform = nn.Linear(3*args.hidden_units, args.hidden_units, bias=False)
        self.linear_transform = nn.Linear(args.gnn_hidden_units, args.gnn_hidden_units, bias=False)

        self.gnndrop = nn.Dropout(args.dropout_rate)

        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.args = args
        self.apply(self._init_weights)

        # user-specific gating
        self.gate_item = Variable(torch.zeros(args.gnn_hidden_units, 1).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_user = Variable(torch.zeros(args.gnn_hidden_units, args.maxlen).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_item = torch.nn.init.xavier_uniform_(self.gate_item)
        self.gate_user = torch.nn.init.xavier_uniform_(self.gate_user)


    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.args.gnn_hidden_units)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def gnn_encode(self, items, args):

        if args.sample_type == 'sparse':
            sparse_edge_index = SparseTensor(row=self.global_graph.edge_index[0],
                                        col=self.global_graph.edge_index[1],
                                        sparse_sizes=self.global_graph.size())
            subgraph_loaders = NeighborSampler(sparse_edge_index, node_idx=items, sizes=self.args.sample_size,
                                            shuffle=False,
                                            num_workers=0, batch_size=items.shape[0])
        else:
            subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.sample_size,
                                           shuffle=False,
                                           num_workers=0, batch_size=items.shape[0])

        g_adjs = []
        s_nodes = []
        for (b_size, node_idx, adjs) in subgraph_loaders:
            if isinstance(adjs, list):
                g_adjs.extend([adj.to(self.device) for adj in adjs])
            else:
                g_adjs.append(adjs.to(self.device))
            n_idxs = node_idx.to(self.device)
            s_nodes = self.item_embeddings(n_idxs).squeeze()

        g_hidden = self.global_gnn(s_nodes, g_adjs, self.global_graph.edge_attr, args.sample_type)

        return g_hidden

    def final_att_net(self, seq_mask, hidden):
        batch_size = hidden.shape[0]
        lens = hidden.shape[1]
        pos_emb = self.position_embeddings.weight[:lens]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hidden_result = torch.zeros_like(hidden)
        
        for i in range(lens):
            current_hidden = hidden[:, :i+1, :]  # Current segment of hidden up to i
            current_mask = seq_mask[:, :i+1]  # Corresponding segment of mask
            current_pos_emb = pos_emb[:, :i+1, :]  # Corresponding segment of position embeddings

            mask_sum = torch.sum(current_mask, 1)
            seq_hidden = torch.sum(current_hidden * current_mask, -2) / (mask_sum + (mask_sum == 0).float())
            seq_hidden[mask_sum.squeeze() == 0] = 0

            seq_hidden = seq_hidden.unsqueeze(1).repeat(1, i+1, 1)
            
            item_hidden = torch.matmul(torch.cat([current_pos_emb, current_hidden], -1), self.w_1)
            item_hidden = torch.tanh(item_hidden)
            score = torch.sigmoid(self.linear_1(item_hidden) + self.linear_2(seq_hidden))
            att_score = torch.matmul(score, self.w_2)
            att_score_masked = att_score * current_mask
            output = torch.sum(att_score_masked * current_hidden, 1)
            hidden_result[:, i, :] = output
        return hidden_result

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').            Unmasked positions are filled with float(0.0).        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -10000.0).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, user_ids, inputs, pos_seqs, neg_seqs, args):

        seq = inputs.flatten()
        seq_mask = (inputs == 0).float().unsqueeze(-1)
        seq_mask = 1.0 - seq_mask

        seq_hidden_global_a = self.gnn_encode(seq, args).view(-1, self.args.maxlen, self.args.gnn_hidden_units)

        user_emb = self.user_embeddings(user_ids).view(-1, self.args.gnn_hidden_units)

        gating_score_a = torch.sigmoid(torch.matmul(seq_hidden_global_a, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)

        user_seq_a = self.gnndrop(user_seq_a)
        hidden = self.linear_transform(user_seq_a)

        seq_out = self.final_att_net(seq_mask, hidden)
        seq_out = self.dropout(seq_out)
    

        teacher_pos_embs = self.item_embeddings(pos_seqs)
        teacher_neg_embs = self.item_embeddings(neg_seqs)
        teacher_pos_logits = (seq_out * teacher_pos_embs).sum(dim=-1)
        teacher_neg_logits = (seq_out * teacher_neg_embs).sum(dim=-1)
        return teacher_pos_logits, teacher_neg_logits



    def predict(self, user_ids, inputs, item_indices):

        seq = inputs.flatten()
        seq_mask = (inputs == 0).float().unsqueeze(-1)
        seq_mask = 1.0 - seq_mask
        seq_hidden_global_a = self.gnn_encode(seq, args=self.args).view(-1, self.args.maxlen, self.args.gnn_hidden_units)
        user_emb = self.user_embeddings(user_ids).view(-1, self.args.gnn_hidden_units)
        gating_score_a = torch.sigmoid(torch.matmul(seq_hidden_global_a, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)
        hidden = self.linear_transform(user_seq_a)
        seq_out = self.final_att_net(seq_mask, hidden)

        final_feat = seq_out[:, -1, :] # only use last QKV classifier, a waste
        
        item_embs = self.item_embeddings(torch.LongTensor(item_indices).to(self.device)) # (U, I, C)
        logits = torch.bmm(item_embs, final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
