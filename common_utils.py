import copy
import random
import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData


def graph_generator(op_feature, mc_feature, relation, pair_feature, verbose=True):
    """
    Build the bipartite op <-> machine graph of a whole batch as a single HeteroData
    object (vectorized, no per-instance Python loop).

    Args:
        op_feature:   [B, N, d_o] - operation embeddings (N = J*O)
        mc_feature:   [B, M, d_m] - machine embeddings
        relation:     [B, N, M]   - whether an operation may still run on a machine (True=yes)
        pair_feature: [B, N, M, d_p] - edge features (0 for pairs that are not connected)
        verbose:      print diagnostics about empty / all-zero edges

    Returns:
        A single HeteroData object representing the batch of graphs.
    """
    device = op_feature.device
    B, N, d_o = op_feature.shape  # N = J*O
    _, M, d_m = mc_feature.shape

    relation_bool = relation.bool()

    op_has_connection = relation_bool.any(dim=2)   # [B, N]
    mch_has_connection = relation_bool.any(dim=1)  # [B, M]

    edge_mask = relation_bool & op_has_connection.unsqueeze(2) & mch_has_connection.unsqueeze(1)
    edge_indices = edge_mask.nonzero(as_tuple=False)

    if edge_indices.numel() == 0:
        if verbose:
            print("Warning: The entire batch has no edges!")
        hd = HeteroData()
        hd['op'].x = op_feature.view(B * N, d_o)
        hd['mch'].x = mc_feature.view(B * M, d_m)
        empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
        hd['op', 'assign', 'mch'].edge_index = empty_index
        hd['mch', 'recruit', 'op'].edge_index = empty_index
        hd['op'].batch = torch.arange(B, device=device).repeat_interleave(N)
        hd['mch'].batch = torch.arange(B, device=device).repeat_interleave(M)
        return hd

    batch_idx = edge_indices[:, 0]
    op_idx_local = edge_indices[:, 1]
    mch_idx_local = edge_indices[:, 2]

    op_idx_global = op_idx_local + batch_idx * N
    mch_idx_global = mch_idx_local + batch_idx * M

    edge_attr = pair_feature[batch_idx, op_idx_local, mch_idx_local]

    if verbose and (edge_attr.abs().sum(dim=1) == 0).any():
        zero_edges = (edge_attr.abs().sum(dim=1) == 0).sum().item()
        print(f"Warning: Batch has {zero_edges}/{len(edge_attr)} edges with all-zero features")

    hd = HeteroData()

    # node features: [B, N, d] -> [B*N, d]
    hd['op'].x = op_feature.reshape(-1, d_o)
    hd['mch'].x = mc_feature.reshape(-1, d_m)

    hd['op'].batch = torch.arange(B, device=device).repeat_interleave(N)
    hd['mch'].batch = torch.arange(B, device=device).repeat_interleave(M)

    hd['op', 'assign', 'mch'].edge_index = torch.stack([op_idx_global, mch_idx_global], dim=0)
    hd['op', 'assign', 'mch'].edge_attr = edge_attr

    hd['mch', 'recruit', 'op'].edge_index = torch.stack([mch_idx_global, op_idx_global], dim=0)
    hd['mch', 'recruit', 'op'].edge_attr = edge_attr
    return hd


"""
    action selection
"""


def sample_action(p):
    idx = torch.multinomial(p, num_samples=1).squeeze(1)
    logp = torch.log(p[torch.arange(p.size(0)), idx])
    return idx, logp


def replay_action(p, idx):
    sel = p[torch.arange(p.size(0), device=p.device), idx]
    return torch.log(sel)


def greedy_select_action(p):
    _, index = torch.max(p, dim=1)
    return index


"""
    heuristic (PDR) baselines
"""


def min_element_index(array):
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_mch_list_for_job(chosen_job, env):
    env_idx = 0
    n_m = env.number_of_machines

    chosen_op = env.candidate[env_idx, chosen_job]

    compatible_mchs_mask = ~env.candidate_process_relation[env_idx, chosen_job, :]
    if not np.any(compatible_mchs_mask):
        return np.array([], dtype=int)

    job_ready_time = env.candidate_free_time[env_idx, chosen_job]  # when the job becomes free
    all_mch_ready = env.mch_free_time[env_idx, :]                  # when each machine becomes free

    last_mch_from_job = env.job_last_mch[env_idx, chosen_job]      # previous machine of this job
    all_last_ops_on_mch = env.mch_last_op[env_idx, :]              # previous operation of each machine

    norm_tts = np.zeros(n_m, dtype=float)
    if last_mch_from_job != -1:
        norm_tts = env.trans_time[env_idx, last_mch_from_job, :]

    norm_sts = np.zeros(n_m, dtype=float)
    valid_st_mask = (all_last_ops_on_mch != -1)
    if np.any(valid_st_mask):
        last_ops = all_last_ops_on_mch[valid_st_mask]
        mch_ids = np.nonzero(valid_st_mask)[0]
        norm_sts[valid_st_mask] = env.setup_time[env_idx, last_ops, chosen_op, mch_ids]

    pot_job_arrival = job_ready_time + norm_tts
    pot_mch_ready = all_mch_ready + norm_sts

    est_values = np.maximum(pot_job_arrival, pot_mch_ready)
    est_values[~compatible_mchs_mask] = float('inf')

    min_est = np.min(est_values)
    if not np.isfinite(min_est):
        return np.array([], dtype=int)

    return np.where(est_values == min_est)[0]


def eft_mch_list_for_job(chosen_job, env):
    env_idx = 0
    n_m = env.number_of_machines

    chosen_op = env.candidate[env_idx, chosen_job]

    compatible_mchs_mask = ~env.candidate_process_relation[env_idx, chosen_job, :]
    if not np.any(compatible_mchs_mask):
        return np.array([], dtype=int)

    job_ready_time = env.candidate_free_time[env_idx, chosen_job]   # scalar (normalized)
    all_mch_ready = env.mch_free_time[env_idx, :]                   # [M]

    all_proc_times = np.array(env.op_pt.filled(0.0)[env_idx, chosen_op, :], copy=False)  # [M]
    trans_mat = env.trans_time                                      # [E, M, M]
    setup_tensor = env.setup_time                                   # [E, N, N, M]

    last_mch_from_job = env.job_last_mch[env_idx, chosen_job]  # scalar (-1 if none)
    all_last_ops_on_mch = env.mch_last_op[env_idx, :]          # [M] (-1 if none)


    norm_tts = np.zeros(n_m, dtype=float)
    if last_mch_from_job != -1:
        norm_tts = trans_mat[env_idx, last_mch_from_job, :]  # [M]

    norm_sts = np.zeros(n_m, dtype=float)
    valid_st_mask = (all_last_ops_on_mch != -1)                # [M]
    if np.any(valid_st_mask):
        last_ops = all_last_ops_on_mch[valid_st_mask]          # [n_valid]
        mch_ids = np.nonzero(valid_st_mask)[0]                 # [n_valid]
        # setup_tensor[e, from_op, to_op, m]
        norm_sts[valid_st_mask] = setup_tensor[env_idx, last_ops, chosen_op, mch_ids]

    pot_job_ready = job_ready_time + norm_tts          # [M]
    pot_mch_ready = all_mch_ready + norm_sts           # [M]
    pot_start = np.maximum(pot_job_ready, pot_mch_ready)  # [M]

    pot_finish = pot_start + all_proc_times            # [M]

    pot_finish[~compatible_mchs_mask] = float('inf')
    min_eft = np.min(pot_finish)
    if not np.isfinite(min_eft):
        return np.array([], dtype=int)

    return np.where(pot_finish == min_eft)[0]


def heuristic_select_action(method, mch_selection_rule, env):
    chosen_job = -1
    chosen_mch = -1

    job_state = (env.mask[0] == 0)

    process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    job_state = process_job_state & job_state

    available_jobs = np.where(job_state == True)[0]
    available_ops = env.candidate[0][available_jobs]

    if method == 'FIFO':
        # selecting the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_jobs]
        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        chosen_job = np.random.choice(chosen_job_list)

        # select the earliest ready machine
        if mch_selection_rule == 'EST':
            mch_state = ~env.candidate_process_relation[0, chosen_job]
            available_mchs = np.where(mch_state == True)[0]
            mch_free_time = env.mch_free_time[0][available_mchs]
            chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
        elif mch_selection_rule == 'EFT':
            chosen_mch_list = eft_mch_list_for_job(chosen_job, env)

        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        chosen_job = np.random.choice(chosen_job_list)

        if mch_selection_rule == 'EST':
            chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        elif mch_selection_rule == 'EFT':
            chosen_mch_list = eft_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]

        action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

        # select a machine which can immediately process the chosen job
        if mch_selection_rule == 'EST':
            chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        elif mch_selection_rule == 'EFT':
            chosen_mch_list = eft_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()

    action = chosen_job * env.number_of_machines + chosen_mch
    return action


"""
    misc
"""

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
