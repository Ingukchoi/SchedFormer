from dataclasses import dataclass
import sys

import numpy as np
import numpy.ma as ma
import torch

from params import configs


@dataclass
class EnvState:
    """
    Let E/N/J/M denote the number of envs/operations/jobs/machines.

        fea_j_tensor            [E, N, 2]     operation raw features
        fea_m_tensor            [E, M, 4]     machine raw features
        fea_pairs_tensor        [E, J, M, 3]  raw features of the (candidate op, machine) pairs
        fea_all_pairs_tensor    [E, N, M, 3]  raw features of every (op, machine) pair
        op_full_mask_tensor     [E, N]        True for operations already removed from O_u
        mc_full_mask_tensor     [E, M]        True for machines that still have work left
        process_relation_tensor [E, N, M]     True for op-machine pairs that are still alive
        dynamic_pair_mask_tensor[E, J, M]     True for infeasible (candidate op, machine) pairs
        candidate_tensor        [E, J]        index of the current candidate operation of each job
    """
    fea_j_tensor: torch.Tensor = None
    op_full_mask_tensor: torch.Tensor = None
    mc_full_mask_tensor: torch.Tensor = None
    process_relation_tensor: torch.Tensor = None

    fea_m_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None
    candidate_tensor: torch.Tensor = None
    fea_pairs_tensor: torch.Tensor = None
    fea_all_pairs_tensor: torch.Tensor = None

    device = torch.device(configs.device)

    def update(self, fea_j, op_full_mask, mc_full_mask, process_relation, fea_all_pairs,
               fea_m, dynamic_pair_mask, candidate, fea_pairs):
        device = self.device
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device)
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)
        self.fea_all_pairs_tensor = torch.from_numpy(np.copy(fea_all_pairs)).float().to(device)

        self.op_full_mask_tensor = torch.from_numpy(np.copy(op_full_mask)).to(device)
        self.mc_full_mask_tensor = torch.from_numpy(np.copy(mc_full_mask)).to(device)
        self.process_relation_tensor = torch.from_numpy(np.copy(process_relation)).to(device)

        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)


class MultiFJSSPEnv:
    def __init__(self, n_j, n_m):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.state = EnvState()

    def set_static_properties(self):
        self.env_idxs = np.arange(self.number_of_envs)
        self.op_idx = np.arange(self.number_of_ops)[np.newaxis, :]

    def set_initial_data(self, job_length_list, op_pt_list, ready_time_list, trans_time_list,
                         setup_time_list, maint_k_list, maint_time_list):

        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.op_pt = np.array(op_pt_list)
        self.number_of_ops = self.op_pt.shape[1]
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        self.set_static_properties()

        # ---------------- Processing Time ----------------
        self.true_op_pt = np.copy(self.op_pt)
        self.pt_upper_bound = np.max(self.true_op_pt)
        self.op_pt = self.true_op_pt / self.pt_upper_bound

        # ---------------- Ready Time (R) ----------------
        self.true_ready_time = np.array(ready_time_list)
        self.ready_time = self.true_ready_time / self.pt_upper_bound

        # ---------------- Transportation Time (T) ----------------
        self.true_trans_time = np.array(trans_time_list)
        self.trans_time = self.true_trans_time / self.pt_upper_bound

        # ---------------- Setup Time (S) ----------------
        self.true_setup_time = np.array(setup_time_list)
        self.setup_time = self.true_setup_time / self.pt_upper_bound

        # ---------------- Maintenance (M) ----------------
        self.true_maint_k = np.array(maint_k_list, dtype=int)          # <= 0 means no maintenance
        self.true_maint_time = np.array(maint_time_list, dtype=float)
        self.maint_time = self.true_maint_time / self.pt_upper_bound

        self.process_relation = (self.true_op_pt != 0)
        self.reverse_process_relation = ~self.process_relation

        self.compatible_mch = np.sum(self.process_relation, 1)

        self.unmasked_op_pt = np.copy(self.op_pt)  # normalized, 0 for incompatible pairs

        head_op_id = np.zeros((self.number_of_envs, 1))

        self.job_first_op_id = np.concatenate(
            [head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype('int')
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        self.initial_vars()  # Must be called after new data is loaded

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)


        self.op_mean_pt = np.mean(self.op_pt, axis=2).data
        self.op_min_pt = np.min(self.op_pt, axis=-1).data

        # job remaining number of operations
        self.op_match_job_left_op_nums = np.array([np.repeat(self.job_length[k],
                                                             repeats=self.job_length[k])
                                                   for k in range(self.number_of_envs)])
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])

        self.op_match_job_remain_work = np.array([np.repeat(self.job_remain_work[k], repeats=self.job_length[k])
                                                  for k in range(self.number_of_envs)])

        self.construct_op_features()

        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        self.mc_full_mask = torch.from_numpy(self.mch_current_available_op_nums > 0).to(torch.bool)

        self.candidate_pt = np.array([self.unmasked_op_pt[k][self.candidate[k]]
                                      for k in range(self.number_of_envs)])

        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)

        self.construct_mch_features()
        self.construct_pair_features()
        self.construct_all_pair_features()

        self.state.update(self.fea_j, self.deleted_op_nodes, self.mc_full_mask, self.process_relation,
                          self.fea_all_pairs, self.fea_m, self.dynamic_pair_mask,
                          self.candidate, self.fea_pairs)
        return self.state

    def initial_vars(self):
        self.step_count = 0
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        self.op_ct = np.zeros((self.number_of_envs, self.number_of_ops))

        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))
        self.next_schedule_time = np.zeros(self.number_of_envs)

        self.candidate_free_time = np.copy(self.ready_time)            # [E, J] (normalized)
        self.true_candidate_free_time = np.copy(self.true_ready_time)  # [E, J] (true)

        self.true_op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = np.copy(self.job_first_op_id)

        # mask[i,j] : whether the jth job of ith env is scheduled (have no unscheduled operations)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.number_of_ops))

        # the last machine used by a job [E, J] / the last operation processed by a machine [E, M]
        self.job_last_mch = np.full((self.number_of_envs, self.number_of_jobs), -1, dtype=int)
        self.mch_last_op = np.full((self.number_of_envs, self.number_of_machines), -1, dtype=int)
        self.mch_ops_since_maint = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)

        # earliest start time of each (candidate, machine) pair, [E, J, M]
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)  # [E, J, 1]
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)              # [E, 1, M]
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)
        self.pair_free_after_const_time = np.copy(self.pair_free_time)

        self.alive_process_relation = np.copy(self.process_relation)

        # mask[i,j] : whether the jth op of ith env is deleted (from the set $O_u$)
        self.deleted_op_nodes = np.full(shape=(self.number_of_envs, self.number_of_ops),
                                        fill_value=0, dtype=bool)

    def _get_trans_time_pairs(self, normalized=True):
        tt_data = self.trans_time if normalized else self.true_trans_time
        tt_pairs = np.zeros((self.number_of_envs, self.number_of_jobs, self.number_of_machines))

        # Find valid (env, job) pairs that have a predecessor
        valid_mask = (self.job_last_mch != -1)  # [E, J]
        env_idx, job_idx = np.where(valid_mask)

        if env_idx.size > 0:
            last_mch_idx = self.job_last_mch[valid_mask]  # [n_valid]
            # For each valid (env, job), get trans time from last_mch to ALL machines
            tt_pairs[env_idx, job_idx, :] = tt_data[env_idx, last_mch_idx, :]

        return tt_pairs

    def _get_setup_time_pairs(self, normalized=True):
        st_data = self.setup_time if normalized else self.true_setup_time
        st_pairs = np.zeros((self.number_of_envs, self.number_of_jobs, self.number_of_machines))

        env_idx, job_idx, mch_idx = np.indices(st_pairs.shape)      # [E,J,M] each
        last_ops = self.mch_last_op[env_idx, mch_idx]               # [E,J,M]
        cand_ops = self.candidate[env_idx, job_idx]                 # [E,J,M]

        valid_mask = (last_ops != -1)
        if valid_mask.any():
            env_f = env_idx[valid_mask]
            last_f = last_ops[valid_mask]
            cand_f = cand_ops[valid_mask]
            mch_f = mch_idx[valid_mask]
            # 4D indexing: [E, N, N, M]
            st_pairs[valid_mask] = st_data[env_f, last_f, cand_f, mch_f]

        return st_pairs

    def _maintenance_due_mask(self, env_ids, mch_ids):
        k = self.true_maint_k[env_ids, mch_ids]
        cnt = self.mch_ops_since_maint[env_ids, mch_ids]

        valid = (k > 0) & (cnt > 0)
        due = np.zeros_like(valid, dtype=bool)

        if np.any(valid):
            # only take the modulo where k>0, which avoids a divide-by-zero warning
            due[valid] = (np.mod(cnt[valid], k[valid]) == 0)

        return due

    def step(self, actions):
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.env_idxs, chosen_job]

        if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
            print(
                f'SFEnv.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # --- Calculate Setup and Transportation Times ---
        last_mch_from_job = self.job_last_mch[self.env_idxs, chosen_job]
        last_op_on_mch = self.mch_last_op[self.env_idxs, chosen_mch]

        norm_tt = np.zeros(self.number_of_envs)
        norm_st = np.zeros(self.number_of_envs)
        true_tt = np.zeros(self.number_of_envs)
        true_st = np.zeros(self.number_of_envs)

        # Transportation Time (Job-based)
        valid_tt_mask = (last_mch_from_job != -1)
        if valid_tt_mask.any():
            env_idx_tt = self.env_idxs[valid_tt_mask]
            last_mch_tt = last_mch_from_job[valid_tt_mask]
            chosen_mch_tt = chosen_mch[valid_tt_mask]

            norm_tt[valid_tt_mask] = self.trans_time[env_idx_tt, last_mch_tt, chosen_mch_tt]
            true_tt[valid_tt_mask] = self.true_trans_time[env_idx_tt, last_mch_tt, chosen_mch_tt]

        # Setup Time (Machine-based)
        valid_st_mask = (last_op_on_mch != -1)
        if valid_st_mask.any():
            env_idx_st = self.env_idxs[valid_st_mask]
            last_op_st = last_op_on_mch[valid_st_mask]
            chosen_op_st = chosen_op[valid_st_mask]
            chosen_mch_st = chosen_mch[valid_st_mask]

            norm_st[valid_st_mask] = self.setup_time[
                env_idx_st, last_op_st, chosen_op_st, chosen_mch_st
            ]
            true_st[valid_st_mask] = self.true_setup_time[
                env_idx_st, last_op_st, chosen_op_st, chosen_mch_st
            ]

        # start time = max(job arrives after transport, machine ready after setup)
        norm_job_ready = self.candidate_free_time[self.env_idxs, chosen_job] + norm_tt
        norm_mch_ready = self.mch_free_time[self.env_idxs, chosen_mch] + norm_st
        chosen_op_st = np.maximum(norm_job_ready, norm_mch_ready)

        true_job_ready = self.true_candidate_free_time[self.env_idxs, chosen_job] + true_tt
        true_mch_ready = self.true_mch_free_time[self.env_idxs, chosen_mch] + true_st
        true_chosen_op_st = np.maximum(true_job_ready, true_mch_ready)

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job])
        self.candidate[self.env_idxs, chosen_job] += candidate_add_flag
        self.mask[self.env_idxs, chosen_job] = (1 - candidate_add_flag)

        # --- Update Completion Times ---
        self.op_ct[self.env_idxs, chosen_op] = chosen_op_st + self.op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[self.env_idxs, chosen_op]
        self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[self.env_idxs, chosen_op]

        self.true_op_ct[self.env_idxs, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
            self.env_idxs, chosen_op]
        self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
            self.env_idxs, chosen_op]

        # --- Maintenance: a machine is serviced after every k operations ---
        self.mch_ops_since_maint[self.env_idxs, chosen_mch] += 1
        due_mask = self._maintenance_due_mask(self.env_idxs, chosen_mch)
        if due_mask.any():
            env_idx_due = self.env_idxs[due_mask]
            mch_due = chosen_mch[due_mask]

            self.mch_free_time[env_idx_due, mch_due] += self.maint_time[env_idx_due, mch_due]
            self.true_mch_free_time[env_idx_due, mch_due] += self.true_maint_time[env_idx_due, mch_due]

            # the machine starts a new maintenance cycle
            self.mch_ops_since_maint[env_idx_due, mch_due] = 0

        # update the dynamic state trackers used by the setup / transport terms
        self.job_last_mch[self.env_idxs, chosen_job] = chosen_mch
        self.mch_last_op[self.env_idxs, chosen_mch] = chosen_op

        self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[
            self.env_idxs, chosen_op])

        # update the candidate message
        mask_temp = candidate_add_flag
        self.candidate_pt[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_pt[mask_temp, chosen_op[mask_temp] + 1]
        self.candidate_process_relation[mask_temp, chosen_job[mask_temp]] = \
            self.reverse_process_relation[mask_temp, chosen_op[mask_temp] + 1]
        self.candidate_process_relation[~mask_temp, chosen_job[~mask_temp]] = 1

        # compute the next schedule time
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)  # [E, J, 1]
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)              # [E, 1, M]


        tt_pairs = self._get_trans_time_pairs(normalized=True)  # [E, J, M]
        st_pairs = self._get_setup_time_pairs(normalized=True)  # [E, J, M]

        self.pair_free_after_const_time = np.maximum(candidateFT_for_compare + tt_pairs,
                                                     mchFT_for_compare + st_pairs)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)
        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

        self.op_scheduled_flag[self.env_idxs, chosen_op] = 1

        self.deleted_op_nodes = \
            np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.op_scheduled_flag)

        # keep only the op-mch edge that is currently being processed
        self.alive_process_relation[self.env_idxs, chosen_op, :] = False
        self.alive_process_relation[self.env_idxs, chosen_op, chosen_mch] = True

        finished_mask = self.deleted_op_nodes  # [E, N] bool
        if finished_mask.any():
            self.alive_process_relation[finished_mask] = False

        # update operation raw features
        job_op_span = (self.op_idx >= (self.job_first_op_id[self.env_idxs, chosen_job])[:, np.newaxis]) & \
                      (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:, np.newaxis])
        self.op_match_job_left_op_nums[job_op_span] -= 1
        self.op_match_job_remain_work[job_op_span] -= \
            np.tile(self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis], (1, self.number_of_ops))[job_op_span]

        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)
        self.construct_op_features()

        # update dynamic pair mask
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        # update machine raw features
        self.mch_current_available_op_nums -= self.process_relation[
            self.env_idxs, chosen_op]

        self.mc_full_mask = torch.from_numpy(self.mch_current_available_op_nums > 0).to(torch.bool)
        mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        self.construct_mch_features()
        self.construct_pair_features()
        self.construct_all_pair_features()

        self.state.update(self.fea_j, self.deleted_op_nodes, self.mc_full_mask,
                          self.alive_process_relation, self.fea_all_pairs, self.fea_m,
                          self.dynamic_pair_mask, self.candidate, self.fea_pairs)

        return self.state, self.done()

    def done(self):
        return np.ones(self.number_of_envs) * (self.step_count >= self.number_of_ops)

    def construct_op_features(self):
        scheduled = self.op_scheduled_flag.astype(bool)      # [E, N]
        op_remain_pt = np.where(scheduled, self.op_remain_work, self.op_min_pt)

        self.fea_j = np.stack((self.op_scheduled_flag,
                               op_remain_pt
                               ), axis=2)

    def _mch_maint_ops_left(self):
        k = self.true_maint_k
        cnt = self.mch_ops_since_maint
        ops_left = np.full_like(cnt, fill_value=-1, dtype=float)

        has_maint = (k > 0)
        if has_maint.any():
            k_eff = k[has_maint].astype(float)
            cnt_eff = cnt[has_maint].astype(float)
            ops_left[has_maint] = np.where(cnt_eff == 0, k_eff, k_eff - np.mod(cnt_eff, k_eff))
        return ops_left

    def construct_mch_features(self):
        mch_maint_ops_left = self._mch_maint_ops_left()  # [E, M]

        self.fea_m = np.stack((
            self.mch_remain_work,
            self.mch_working_flag,
            mch_maint_ops_left,
            self.maint_time), axis=2)

    def construct_pair_features(self):
        mask = self.dynamic_pair_mask                 # [E,J,M]
        pfa = self.pair_free_after_const_time         # [E,J,M]

        base = self.next_schedule_time[:, None, None]
        est = pfa - base                               # always non-negative
        pt = self.candidate_pt.copy()                  # [E,J,M]
        eet = est + pt

        pt[mask] = 0.0
        est[mask] = -1.0
        eet[mask] = -1.0

        self.fea_pairs = np.stack((pt, est, eet), axis=-1)

    def construct_all_pair_features(self):
        rpr = self.alive_process_relation             # [E,N,M]

        op_pt_ma = ma.array(self.op_pt, mask=~rpr)    # [E,N,M] 
        f_pt = op_pt_ma.filled(0.0)                   # [E,N,M]

        pfa = self.pair_free_after_const_time                        # [E,J,M]
        base = self.next_schedule_time[:, None, None]                # [E,1,1]
        est_jm = pfa - base                                          # [E,J,M]
        eft_jm = est_jm + self.candidate_pt                          # [E,J,M]

        E, J = self.number_of_envs, self.number_of_jobs
        env_idx, job_idx = np.indices((E, J))                        # [E,J]
        op_idx = self.candidate[env_idx, job_idx]                    # [E,J]

        job_active = ~np.all(self.dynamic_pair_mask, axis=2)         # [E,J]

        est_full = np.full((E, self.number_of_ops, self.number_of_machines), -1.0, dtype=f_pt.dtype)
        eft_full = np.full_like(est_full, -1.0)

        if job_active.any():
            env_f = env_idx[job_active]
            op_f = op_idx[job_active]
            est_full[env_f, op_f, :] = est_jm[job_active, :]
            eft_full[env_f, op_f, :] = eft_jm[job_active, :]

        self.fea_all_pairs = np.stack([f_pt, est_full, eft_full], axis=-1)  # [E,N,M,3]
        self.fea_all_pairs[~rpr, 0] = 0.0
