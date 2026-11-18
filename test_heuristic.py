import os
import sys
import time

import numpy as np
from tqdm import tqdm

from common_utils import setup_seed, heuristic_select_action
from params import configs
from SFEnv import MultiFJSSPEnv

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


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


def test_heuristic_on_instance(instance, heuristic, mch_heuristic, seed):
    setup_seed(seed)

    n_j = instance["job_length"].shape[0]
    n_m = instance["op_pt"].shape[1]

    env = MultiFJSSPEnv(n_j=n_j, n_m=n_m)
    env.set_initial_data(
        [instance["job_length"]], [instance["op_pt"]],
        [instance["ready_time"]], [instance["trans_time"]], [instance["setup_time"]],
        [instance["maint_k"]], [instance["maint_time"]]
    )

    t1 = time.time()
    while True:
        action = heuristic_select_action(heuristic, mch_heuristic, env)
        _, done = env.step(np.array([action]))
        if done.all():
            break
    t2 = time.time()

    return np.array([[env.current_makespan[0], t2 - t1]])


def test_heuristic_dir(npz_dir, heuristic, mch_heuristic, seed):
    file_list = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    if not file_list:
        print(f"No test files under: {npz_dir}")
        return None

    print("-" * 25 + "Test Heuristic Methods" + "-" * 25)
    print(f"✓ Directory: {npz_dir}")
    print(f"✓ Heuristic: {heuristic}x{mch_heuristic}")

    all_results = []  # [makespan, time] of each instance
    for fname in tqdm(file_list, file=sys.stdout, desc="progress", colour='blue'):
        inst = load_instance_npz(os.path.join(npz_dir, fname))
        res = test_heuristic_on_instance(inst, heuristic, mch_heuristic, seed)
        all_results.append(res[0])

    return np.array(all_results)  # [N_inst, 2]


def main():
    setup_seed(configs.seed_test)

    problem = configs.test_problem
    instance_size = configs.test_size

    npz_dir = f'./data/Synthetic/{problem}/{instance_size}'
    if not os.path.isdir(npz_dir):
        print(f"[ERROR] Invalid npz_dir: {npz_dir}")
        return

    save_root = f'./test_results/heuristics/{problem}/{instance_size}'
    os.makedirs(save_root, exist_ok=True)

    for method in configs.test_method:
        for mch_method in configs.mch_method:
            save_path = os.path.join(
                save_root, f'{problem}_{instance_size}_Result_{method}x{mch_method}.npy')
            if not configs.cover_heu_flag and os.path.exists(save_path):
                print(f"[Skip] Exists: {save_path}")
                continue

            results = test_heuristic_dir(
                npz_dir=npz_dir,
                heuristic=method,
                mch_heuristic=mch_method,
                seed=configs.seed_test
            )
            if results is None:
                continue

            print(f"testing results of {problem} : {method}x{mch_method}_{instance_size}:")
            print(f"makespan({method}): {results[:, 0].mean()}")
            print(f"time:              {results[:, 1].mean()}")
            print("-" * 34 + "-" * 34)
            print("")
            np.save(save_path, results)


if __name__ == '__main__':
    main()
