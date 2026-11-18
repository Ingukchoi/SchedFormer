import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import graph_generator
from model.attention_layer import MixedCrossAttention, EdgeUpdater, \
    MultiheadSelfAttention_for_OP_withRoPE, MultiheadSelfAttention_for_MC
from model.sub_layers import MLP, RMSNorm, SwiGLU


class ScheduleFormerLayer(nn.Module):
    def __init__(self, config):
        super(ScheduleFormerLayer, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.fea_e_input_dim = config.fea_e_input_dim

        self.num_layers = config.num_layer
        self.hidden_dim = config.hidden_dim
        self.num_head = config.multi_head
        self.dropout_prob = config.dropout_prob

        self.op_self_attention_blocks = torch.nn.ModuleList()
        self.mc_self_attention_blocks = torch.nn.ModuleList()
        self.Cross_attention_blocks = torch.nn.ModuleList()

        self.Projection_O = torch.nn.Linear(self.fea_j_input_dim, self.hidden_dim)
        self.Projection_M = torch.nn.Linear(self.fea_m_input_dim, self.hidden_dim)
        self.Projection_E = nn.Linear(self.fea_e_input_dim, self.hidden_dim)

        for _ in range(self.num_layers):
            self.op_self_attention_blocks.append(
                MultiheadSelfAttention_for_OP_withRoPE(self.hidden_dim, self.num_head, self.dropout_prob))

        for _ in range(self.num_layers):
            self.mc_self_attention_blocks.append(
                MultiheadSelfAttention_for_MC(self.hidden_dim, self.num_head, self.dropout_prob))

        for _ in range(self.num_layers):
            self.Cross_attention_blocks.append(MixedCrossAttention(
                d_op=self.hidden_dim,
                d_mch=self.hidden_dim,
                d_edge=self.hidden_dim,
                d_hidden=self.hidden_dim,
                heads=self.num_head,
                dropout=self.dropout_prob))

        self.edge_updaters = nn.ModuleList([EdgeUpdater(self.hidden_dim) for _ in range(self.num_layers)])

        self.RMSNorm_cross_op = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.RMSNorm_cross_mc = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.RMSNorm_self_op = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.RMSNorm_self_mc = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])

        self.RMSNorm_op1 = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.RMSNorm_mc1 = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])

        self.SwiGLU_op1 = nn.ModuleList([SwiGLU(self.hidden_dim, ff_mult=2) for _ in range(self.num_layers)])
        self.SwiGLU_mc1 = nn.ModuleList([SwiGLU(self.hidden_dim, ff_mult=2) for _ in range(self.num_layers)])

        self.RMSNorm_op = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.RMSNorm_mc = nn.ModuleList([RMSNorm(self.hidden_dim) for _ in range(self.num_layers)])

        self.SwiGLU_op = nn.ModuleList([SwiGLU(self.hidden_dim, ff_mult=2) for _ in range(self.num_layers)])
        self.SwiGLU_mc = nn.ModuleList([SwiGLU(self.hidden_dim, ff_mult=2) for _ in range(self.num_layers)])

    def _update_edges_in_graph(self, graph_data, layer, op_x_flat, mch_x_flat):
        e_key = ('op', 'assign', 'mch')
        edge_index = graph_data[e_key].edge_index      # [2, E]
        e_attr = graph_data[e_key].edge_attr           # [E, d]

        src = edge_index[0]  # op global idx
        dst = edge_index[1]  # mch global idx

        x_src = op_x_flat[src]    # [E, d]
        y_dst = mch_x_flat[dst]   # [E, d]

        e_new = self.edge_updaters[layer](e_attr, x_src, y_dst)  # [E, d]

        graph_data[e_key].edge_attr = e_new
        graph_data[('mch', 'recruit', 'op')].edge_attr = e_new

    def forward(self, fea_j, op_full_mask, mc_full_mask, op_mc_relation, fea_all_pairs, fea_m, n_jobs):
        sz_b, M, _ = fea_m.size()
        J = n_jobs
        O = int(fea_j.size(1) / J)

        fea_j = self.Projection_O(fea_j)
        fea_m = self.Projection_M(fea_m)
        fea_e = self.Projection_E(fea_all_pairs)

        for layer in range(self.num_layers):
            ######## pre-stage
            fea_j_norm = self.RMSNorm_cross_op[layer](fea_j)
            fea_m_norm = self.RMSNorm_cross_mc[layer](fea_m)

            if layer != 0:
                graph_data['op'].x = fea_j_norm.contiguous().view(-1, fea_j.size(-1))   # [ΣN, d_hidden]
                graph_data['mch'].x = fea_m_norm.contiguous().view(-1, fea_j.size(-1))

            if layer == 0:
                graph_data = graph_generator(fea_j_norm, fea_m_norm, op_mc_relation, fea_e, False)

            ######## Edge updater
            op_x_flat = fea_j_norm.contiguous().view(-1, self.hidden_dim)   # [B*N, d], N=J*O
            mch_x_flat = fea_m_norm.contiguous().view(-1, self.hidden_dim)  # [B*M, d]
            self._update_edges_in_graph(graph_data, layer, op_x_flat, mch_x_flat)

            ######## Cross-attention Blocks
            out_op, out_mp = self.Cross_attention_blocks[layer](graph_data)

            fea_j = fea_j.view(sz_b, J, O, self.hidden_dim) + out_op.view(sz_b, J, O, self.hidden_dim)
            fea_m = fea_m.view(sz_b, M, self.hidden_dim) + out_mp.view(sz_b, M, self.hidden_dim)

            fea_j = fea_j + self.SwiGLU_op1[layer](self.RMSNorm_op1[layer](fea_j))
            fea_m = fea_m + self.SwiGLU_mc1[layer](self.RMSNorm_mc1[layer](fea_m))

            ######## Self-Attention Blocks
            fea_j = fea_j + self.op_self_attention_blocks[layer](self.RMSNorm_self_op[layer](fea_j), op_full_mask)
            fea_m = fea_m + self.mc_self_attention_blocks[layer](self.RMSNorm_self_mc[layer](fea_m), mc_full_mask)

            fea_j = fea_j + self.SwiGLU_op[layer](self.RMSNorm_op[layer](fea_j))
            fea_m = fea_m + self.SwiGLU_mc[layer](self.RMSNorm_mc[layer](fea_m))  # [ΣM, d_hidden]

        fea_j = fea_j.view(sz_b, J * O, self.hidden_dim)
        return fea_j, fea_m


class ScheduleFormer(nn.Module):
    def __init__(self, config):
        super(ScheduleFormer, self).__init__()
        device = torch.device(config.device)
        self.hidden_dim = config.hidden_dim
        # raw dimension of the op-machine pair features
        self.pair_input_dim = config.fea_e_input_dim

        self.feature_exact = ScheduleFormerLayer(config).to(device)
        self.proj_pair = nn.Linear(self.pair_input_dim, self.hidden_dim)

        self.op_norm = RMSNorm(self.hidden_dim)
        self.mc_norm = RMSNorm(self.hidden_dim)

        self.actor = MLP(config.num_mlp_layers_actor, self.hidden_dim * 2 + self.pair_input_dim,
                         config.hidden_dim_actor, 1).to(device)

    def forward(self, fea_j, op_full_mask, mc_full_mask, op_mc_relation, fea_all_pairs,
                candidate, fea_m, dynamic_pair_mask, fea_pairs):
        sz_b, M = fea_m.size(0), fea_m.size(1)
        J = candidate.size(1)

        fea_j, fea_m = self.feature_exact(fea_j, op_full_mask, mc_full_mask, op_mc_relation,
                                          fea_all_pairs, fea_m, J)

        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_e_serialized = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        Fea_j_JC_serialized = self.op_norm(Fea_j_JC_serialized)
        Fea_m_serialized = self.mc_norm(Fea_m_serialized)

        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_e_serialized), dim=-1)
        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)
        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        return pi
