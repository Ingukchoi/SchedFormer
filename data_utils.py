import os
import random
import sys

import numpy as np

from params import configs


def problem_flags(problem: str):
    """
      'FJSSP'      -> all False (classical FJSP)
      'FJSSP-R'    -> release dates only
      'FJSSP-RS'   -> release dates + sequence-dependent setup times
      'FJSSP-RSMT' -> all four constraints
    """
    suffix = problem.split('-', 1)[1] if '-' in problem else ""

    use_R = 'R' in suffix
    use_S = 'S' in suffix
    use_M = 'M' in suffix
    use_T = 'T' in suffix

    return use_R, use_S, use_M, use_T


def instance_generator(config, n_j, n_m, flag_ready, flag_setup, flag_main, flag_trans):
    """
    Generate one random instance. The FJSP part follows the SD2 protocol of
    "songwenas12/fjsp-drl"; the four extra constraints are sampled on top of it
    and zeroed out when the corresponding flag is False.

    :param config: a package of parameters
    :return:
        job_length : [J]      number of operations of each job
        op_pt      : [N, M]   processing times (0 if incompatible)
        op_per_mch : float    average number of compatible machines per operation
        ready_time : [J]      release dates (50% zeros, others ~ U{1, 10*J})
        trans_time : [M, M]   transportation times ~ U{1,40}, diag = 0
        setup_time : [N, N, M] setup times ~ U{1,50}, diag = 0
        maint_k    : [M]      maintenance period in number of operations (~U{4..8})
        maint_time : [M]      duration of one maintenance ~ U{1,125}
    """
    if config.op_per_job == 0:
        op_per_job = n_m
    else:
        op_per_job = config.op_per_job

    low = config.low
    high = config.high
    data_suffix = config.data_suffix

    op_per_mch_min = 1
    if data_suffix == "nf":  # job shop setting
        op_per_mch_max = 1
    elif data_suffix == "mix":
        op_per_mch_max = n_m
    else:
        op_per_mch_min = config.op_per_mch_min
        op_per_mch_max = config.op_per_mch_max
    if op_per_mch_min < 1 or op_per_mch_max > n_m:
        print(f'Error from Instance Generation: [{op_per_mch_min},{op_per_mch_max}] '
              f'with num_mch : {n_m}')
        sys.exit()

    n_op = int(n_j * op_per_job)
    job_length = np.full(shape=(n_j,), fill_value=op_per_job, dtype=int)
    op_use_mch = np.random.randint(low=op_per_mch_min, high=op_per_mch_max + 1,
                                   size=n_op)

    op_per_mch = float(np.mean(op_use_mch))
    op_pt = np.random.randint(low=low, high=high + 1, size=(n_op, n_m))

    for row in range(op_pt.shape[0]):
        mch_num = int(op_use_mch[row])
        if mch_num < n_m:
            inf_pos = np.random.choice(np.arange(0, n_m), n_m - mch_num, replace=False)
            op_pt[row][inf_pos] = 0

    #############################################
    # ready_time: [J], 50% zeros, others ~ U{1, 10*J}
    ready_time = np.zeros(n_j, dtype=int)
    k = n_j // 2
    if k > 0:
        idx = np.random.choice(n_j, size=k, replace=False)
        ready_time[idx] = np.random.randint(1, 10 * n_j + 1, size=k, dtype=int)

    # trans_time: [M, M], U{1,40}
    trans_time = np.random.randint(1, 40 + 1, size=(n_m, n_m), dtype=int)
    np.fill_diagonal(trans_time, 0)

    # setup_time: [N, N, M], U{1,50}
    setup_time = np.random.randint(1, 50 + 1, size=(n_op, n_op, n_m), dtype=int)
    idx = np.arange(n_op)
    setup_time[idx, idx, :] = 0

    # Machine maintenance
    maint_k = np.random.choice([4, 5, 6, 7, 8], size=n_m)
    maint_time = np.random.randint(1, 125 + 1, size=n_m)
    #############################################

    # Zero out the constraints that are switched off for this problem variant
    #############################################
    if flag_ready == False:
        ready_time = np.zeros_like(ready_time)

    if flag_trans == False:
        trans_time = np.zeros_like(trans_time)

    if flag_setup == False:
        setup_time = np.zeros_like(setup_time)

    if flag_main == False:
        maint_k = np.zeros_like(maint_k) * -1
        maint_time = np.zeros_like(maint_time)
    #############################################

    return job_length, op_pt, op_per_mch, ready_time, trans_time, setup_time, maint_k, maint_time


def generate_data_to_files(seed, directory, config, n_j, n_m, flag_ready, flag_setup, flag_main, flag_trans):
    batch_size = config.data_size

    low = config.low
    high = config.high

    filename = '{}x{}'.format(n_j, n_m)
    np.random.seed(seed)
    random.seed(seed)

    print("-" * 25 + "Data Setting" + "-" * 25)
    print(f"seed : {seed}")
    print(f"data size : {batch_size}")
    print(f"flexibility : {config.data_suffix}")
    print(f"filename : {filename}")
    print(f"processing time : [{low},{high}]")
    print("-" * 50)

    path = os.path.join(directory, filename)
    if (not os.path.exists(path)) or config.cover_data_flag:
        os.makedirs(path, exist_ok=True)

        for idx in range(batch_size):
            dataset_JobLength = []
            dataset_OpPT = []
            dataset_ready = []
            dataset_trans = []
            dataset_setup = []
            dataset_maint_k = []
            dataset_maint_time = []

            job_length, op_pt, _, ready_time, trans_time, setup_time, maint_k, maint_time = \
                instance_generator(config, n_j, n_m, flag_ready, flag_setup, flag_main, flag_trans)

            dataset_JobLength.append(job_length)
            dataset_OpPT.append(op_pt)
            dataset_ready.append(ready_time)
            dataset_trans.append(trans_time)
            dataset_setup.append(setup_time)
            dataset_maint_k.append(maint_k)
            dataset_maint_time.append(maint_time)

            file_path = os.path.join(path, f"{filename}_{idx+1:02d}.npz")

            np.savez(
                file_path,
                dataset_JobLength=dataset_JobLength,
                dataset_OpPT=dataset_OpPT,
                dataset_ready=dataset_ready,
                dataset_trans=dataset_trans,
                dataset_setup=dataset_setup,
                dataset_maint_k=dataset_maint_k,
                dataset_maint_time=dataset_maint_time
            )
    else:
        print("the data already exists...")


def main():
    problem = configs.test_problem
    flag_ready, flag_setup, flag_main, flag_trans = problem_flags(problem)

    generate_data_to_files(configs.seed_datagen,
                           f'./data/Synthetic/{problem}/',
                           configs, configs.n_j, configs.n_m,
                           flag_ready, flag_setup, flag_main, flag_trans)


if __name__ == '__main__':
    main()
    print("success!")
