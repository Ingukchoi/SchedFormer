import os
import random
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW as Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from common_utils import setup_seed, sample_action, replay_action
from data_utils import instance_generator
from model.SchedFormer import ScheduleFormer as Model
from params import configs
from SFEnv import MultiFJSSPEnv

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
device = torch.device(configs.device)

class LinearWarmup(LambdaLR):
    def __init__(self, optimizer, warmup_steps, training_steps, last_epoch=-1):
        def lr_lambda(cur_step):
            if cur_step < warmup_steps:
                return float(cur_step) / float(max(1.0, warmup_steps))
            return max(
                0.0,
                float(training_steps - cur_step) / float(max(1, training_steps - warmup_steps))
            )

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class Trainer:
    def __init__(self, config):
        # seed before the network is built, otherwise the initial weights are drawn
        # from PyTorch's non-deterministically seeded default generator
        setup_seed(config.seed_train)

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.n_j = config.n_j
        self.n_m = config.n_m
        self.low = config.low
        self.high = config.high
        self.config = config
        self.max_updates = config.max_updates
        self.model_store_timestep = config.model_store_timestep
        self.num_envs = config.num_envs
        self.multi_sample = config.multi_sample
        self.accmulation_step = config.accmulation_step
        num_optimizer_steps = self.max_updates // self.accmulation_step
        warmup_steps = int(0.1 * num_optimizer_steps)

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Env
        self.env_rollout = MultiFJSSPEnv(self.n_j, self.n_m)
        self.env_replay = MultiFJSSPEnv(self.n_j, self.n_m)

        # seed
        self.seed_train = config.seed_train

        # Optimizer
        self.model = Model(config)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters (M): {total_params / 1e6:.2f}M")
        print(f"Trainable parameters (M): {trainable_params / 1e6:.2f}M")
        self.optimizer = Optimizer(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay
        )
        self.scheduler = LinearWarmup(self.optimizer, warmup_steps, num_optimizer_steps)
        self.use_amp = (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def imitation_replay_loss(self, dataset_job_length, dataset_op_pt, dataset_ready, dataset_trans,
                              dataset_setup, dataset_maint_k, dataset_maint_time,
                              best_actions_traj, weights):
        """Replay the best trajectories and return their weighted negative log-likelihood."""
        device = next(self.model.parameters()).device
        B, T = best_actions_traj.shape

        state = self.env_replay.set_initial_data(dataset_job_length, dataset_op_pt, dataset_ready,
                                                 dataset_trans, dataset_setup, dataset_maint_k,
                                                 dataset_maint_time)

        total_logprob = 0.0
        total_cnt = 0

        for t in range(T):
            pi = self.model(
                fea_j=state.fea_j_tensor,
                op_full_mask=state.op_full_mask_tensor,
                mc_full_mask=state.mc_full_mask_tensor,
                op_mc_relation=state.process_relation_tensor,
                fea_all_pairs=state.fea_all_pairs_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor
            )

            actions_t = best_actions_traj[:, t]

            logp_t = replay_action(pi, actions_t)  # [B]
            if weights is not None:
                logp_t = logp_t * weights

            total_logprob = total_logprob + logp_t.sum()
            total_cnt += B

            next_state, done = self.env_replay.step(actions=actions_t.cpu().numpy())
            state = next_state

            if done.all():
                break

        replay_makespan = torch.from_numpy(self.env_replay.current_makespan).to(device).reshape(B)
        imitation_loss = - total_logprob / total_cnt
        return imitation_loss, replay_makespan

    def train(self):
        setup_seed(self.seed_train)

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"Multi-task FJSSP: {configs.flag_multi}")
        print("\n")

        if configs.flag_multi == False:
            flag_ready = configs.flag_ready
            flag_setup = configs.flag_setup
            flag_main = configs.flag_maintenance
            flag_trans = configs.flag_trans

        train_score_list = list()

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='red'):

            ep_st = time.time()

            # Task sampling for training
            if configs.flag_multi == True:
                if configs.flag_full == True:
                    flag_ready, flag_setup, flag_main, flag_trans = self.sample_full_task()
                else:
                    flag_ready, flag_setup, flag_main, flag_trans = self.sample_partial_task()

            dataset_job_length, dataset_op_pt, dataset_ready, dataset_trans, dataset_setup, dataset_maint_k, dataset_maint_time, \
                replay_dataset_JobLength, replay_dataset_OpPT, replay_dataset_ready, replay_dataset_trans, replay_dataset_setup, replay_dataset_maint_k, replay_dataset_maint_time \
                = self.sample_training_instances(self.n_j, self.n_m, flag_ready, flag_setup, flag_main, flag_trans)

            state = self.env_rollout.set_initial_data(dataset_job_length, dataset_op_pt, dataset_ready,
                                                      dataset_trans, dataset_setup, dataset_maint_k,
                                                      dataset_maint_time)
            saved_actions = []

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    while True:
                        pi_envs = self.model(fea_j=state.fea_j_tensor,                      # [B, N, 2]
                                             op_full_mask=state.op_full_mask_tensor,        # [B, N]
                                             mc_full_mask=state.mc_full_mask_tensor,        # [B, M]
                                             op_mc_relation=state.process_relation_tensor,  # [B, N, M]
                                             fea_all_pairs=state.fea_all_pairs_tensor,      # [B, N, M, 3]
                                             candidate=state.candidate_tensor,              # [B, J]
                                             fea_m=state.fea_m_tensor,                      # [B, M, 4]
                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [B, J, M]
                                             fea_pairs=state.fea_pairs_tensor               # [B, J, M, 3]
                                             )

                        action_envs, log_p = sample_action(pi_envs)
                        saved_actions.append(action_envs)

                        state, done = self.env_rollout.step(actions=action_envs.cpu().numpy())
                        if done.all():
                            break

                makespan = torch.from_numpy(self.env_rollout.current_makespan).to(device)
                makespan = makespan.reshape(self.num_envs, self.multi_sample)
                actions_T_BP = torch.stack(saved_actions, dim=0)  # [T, B*P]
                T = actions_T_BP.size(0)
                B = self.num_envs
                actions_BPT = actions_T_BP.view(actions_T_BP.size(0), self.num_envs,
                                                self.multi_sample).permute(1, 2, 0).contiguous()

                # keep the best rollout of every instance
                best_p_idx = makespan.argmin(dim=1)

                idx_expanded = best_p_idx.view(B, 1, 1).expand(B, 1, T)          # [B, 1, T]
                best_actions_traj = torch.gather(actions_BPT, 1, idx_expanded)   # [B, 1, T]
                best_actions_traj = best_actions_traj.squeeze(1).contiguous()
                best_actions_traj = best_actions_traj.detach()

                # advantage-style weight: how much better than the average rollout
                mean_ms = makespan.mean(dim=1)          # [B]
                std_ms = makespan.std(dim=1) + 1e-8     # [B]
                best_ms = makespan[torch.arange(B), best_p_idx]  # [B]

                weights = (mean_ms - best_ms) / std_ms  # [B]

            del saved_actions, actions_T_BP, actions_BPT
            del state, pi_envs, action_envs
            del dataset_job_length, dataset_op_pt, dataset_ready, dataset_trans, dataset_setup, dataset_maint_k, dataset_maint_time

            self.model.train()
            if weights is not None:
                weights = weights.to(device)
                weights = weights.detach()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, replay_makespan = self.imitation_replay_loss(
                    replay_dataset_JobLength, replay_dataset_OpPT, replay_dataset_ready,
                    replay_dataset_trans, replay_dataset_setup, replay_dataset_maint_k,
                    replay_dataset_maint_time, best_actions_traj=best_actions_traj, weights=weights)
                loss = loss / self.accmulation_step

            self.scaler.scale(loss).backward()
            Training_score = replay_makespan.mean().item()
            train_score_list.append(Training_score)

            if (i_update + 1) % self.accmulation_step == 0:

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            ep_et = time.time()
            if (i_update + 1) % self.model_store_timestep == 0 or (i_update + 1) == self.max_updates:
                self.save_model(i_update)
            tqdm.write(
                'Episode {}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, Training_score, loss.item() * self.accmulation_step, ep_et - ep_st))

            # Training score recording
            train_ms_log = np.array(train_score_list)
            np.save(os.path.join(configs.log_path, 'Train_score.npy'), train_ms_log)

        # Done.
        print("-" * 25 + "Training Done" + "-" * 25)
        print(f"Training is end. Training results are stored in {configs.log_path} folder.")

    def sample_training_instances(self, n_job, n_mc, flag_ready, flag_setup, flag_main, flag_trans):
        dataset_JobLength = []
        dataset_OpPT = []
        dataset_ready = []
        dataset_trans = []
        dataset_setup = []
        dataset_maint_k = []
        dataset_maint_time = []

        replay_dataset_JobLength = []
        replay_dataset_OpPT = []
        replay_dataset_ready = []
        replay_dataset_trans = []
        replay_dataset_setup = []
        replay_dataset_maint_k = []
        replay_dataset_maint_time = []

        for _ in range(self.num_envs):
            JobLength, OpPT, _, ready_time, trans_time, setup_time, maint_k, maint_time = \
                instance_generator(self.config, n_job, n_mc, flag_ready, flag_setup, flag_main, flag_trans)

            replay_dataset_JobLength.append(JobLength)
            replay_dataset_OpPT.append(OpPT)
            replay_dataset_ready.append(ready_time)
            replay_dataset_trans.append(trans_time)
            replay_dataset_setup.append(setup_time)
            replay_dataset_maint_k.append(maint_k)
            replay_dataset_maint_time.append(maint_time)

            for _ in range(self.multi_sample):
                dataset_JobLength.append(JobLength)
                dataset_OpPT.append(OpPT)
                dataset_ready.append(ready_time)
                dataset_trans.append(trans_time)
                dataset_setup.append(setup_time)
                dataset_maint_k.append(maint_k)
                dataset_maint_time.append(maint_time)
        return dataset_JobLength, dataset_OpPT, dataset_ready, dataset_trans, dataset_setup, dataset_maint_k, dataset_maint_time, \
            replay_dataset_JobLength, replay_dataset_OpPT, replay_dataset_ready, replay_dataset_trans, replay_dataset_setup, replay_dataset_maint_k, replay_dataset_maint_time

    def sample_full_task(self):
        """sample one of the 16 problem variants (each constraint on/off independently)"""
        flag_ready = random.choice([True, False])
        flag_setup = random.choice([True, False])
        flag_main = random.choice([True, False])
        flag_trans = random.choice([True, False])
        return flag_ready, flag_setup, flag_main, flag_trans

    def sample_partial_task(self):
        """sample one of the 4 single-constraint variants"""
        choice = random.randrange(4)

        flag_ready = (choice == 0)
        flag_setup = (choice == 1)
        flag_main = (choice == 2)
        flag_trans = (choice == 3)

        return flag_ready, flag_setup, flag_main, flag_trans

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), configs.log_path + f'/models/checkpoint{epoch+1}.pth')


def main():
    configs.log_path = f"./train_results/{configs.run_name}/FJSSP"
    os.makedirs(configs.log_path + '/models', exist_ok=True)
    trainer = Trainer(configs)
    trainer.train()

if __name__ == '__main__':
    main()
