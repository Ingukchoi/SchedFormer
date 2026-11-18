import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from common_utils import setup_seed, greedy_select_action, sample_action
from model.SchedFormer import ScheduleFormer as Model
from params import configs
from SFEnv import MultiFJSSPEnv

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
device = torch.device(configs.device)


def load_instance_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "job_length": d["dataset_JobLength"][0],
        "op_pt": d["dataset_OpPT"][0],
        "ready_time": d["dataset_ready"][0],
        "trans_time": d["dataset_trans"][0],
        "setup_time": d["dataset_setup"][0],
        "maint_k": d["dataset_maint_k"][0],
        "maint_time": d["dataset_maint_time"][0],
    }


def make_env(instance, replicas):
    n_j = instance["job_length"].shape[0]
    n_m = instance["op_pt"].shape[1]

    env = MultiFJSSPEnv(n_j=n_j, n_m=n_m)
    state = env.set_initial_data(
        [instance["job_length"].copy() for _ in range(replicas)],
        [instance["op_pt"].copy() for _ in range(replicas)],
        [instance["ready_time"].copy() for _ in range(replicas)],
        [instance["trans_time"].copy() for _ in range(replicas)],
        [instance["setup_time"].copy() for _ in range(replicas)],
        [instance["maint_k"].copy() for _ in range(replicas)],
        [instance["maint_time"].copy() for _ in range(replicas)],
    )
    return env, state


def policy(model, state):
    return model(
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


@torch.no_grad()
def test_greedy_strategy(instance, model):
    env, state = make_env(instance, 1)

    t1 = time.time()
    while True:
        pi = policy(model, state)
        action_envs = greedy_select_action(pi)
        state, done = env.step(action_envs.cpu().numpy())
        if done.all():
            break
    t2 = time.time()
    return np.array([[env.current_makespan[0], t2 - t1]])


@torch.no_grad()
def test_sampling_strategy(instance, model, sample_times, seed):
    setup_seed(seed)
    env, state = make_env(instance, sample_times)

    t1 = time.time()
    while True:
        pi = policy(model, state)
        action_envs, _ = sample_action(pi)
        state, done = env.step(action_envs.cpu().numpy())
        if done.all():
            break
    t2 = time.time()
    return np.array([[np.min(env.current_makespan), t2 - t1]])


def main(config):

    flag_sample = config.test_mode
    setup_seed(config.seed_test)
    model_path = f'./train_results/{config.test_model}/FJSSP/models/checkpoint{config.epoch}.pth'
    model = Model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_problem = config.test_problem
    test_size = config.test_size
    dir_path = f"./data/Synthetic/{test_problem}/{test_size}"
    file_list = sorted([f for f in os.listdir(dir_path) if f.endswith(".npz")])

    if not file_list:
        print(f"No test files under: {dir_path}")
        return

    mode_name = "Sampling" if flag_sample else "Greedy"
    print("-" * 25 + "Test Learned Model" + "-" * 25)
    print(f"✓ Problem: {test_problem}")
    print(f"✓ Instance size: {test_size}")
    print(f"✓ Test mode: {mode_name}")
    print(f"✓ # of test instances: {len(file_list)}")
    print("-" * 34 + "-" * 34)

    results = []
    save_path = f'./test_results/{config.test_model}/{test_problem}/{mode_name}'
    os.makedirs(save_path, exist_ok=True)

    for fname in tqdm(file_list, file=sys.stdout, desc="progress",
                      colour='yellow' if not flag_sample else 'red'):
        inst = load_instance_npz(os.path.join(dir_path, fname))

        if not flag_sample:
            res = test_greedy_strategy(inst, model)
        else:
            res = test_sampling_strategy(inst, model, config.sample_times, config.seed_test)
        results.append(res[0])

    results = np.array(results)
    np.save(os.path.join(save_path, f'{test_size}.npy'), results)

    print("testing results:")
    print(f"makespan({mode_name.lower()}): {results[:, 0].mean()}")
    print(f"time: {results[:, 1].mean()}")


if __name__ == '__main__':
    main(configs)
