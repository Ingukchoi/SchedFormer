import argparse


def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser(description='Arguments for SchedFormer')

# args for device
parser.add_argument('--device', type=str, default='cuda', help='Device name')
parser.add_argument('--device_id', type=str, default='0', help='Device id')

# args for instance generation
parser.add_argument('--data_suffix', type=str, default='mix',
                    help="Flexibility setting: 'mix' (1~m machines per operation), "
                         "'nf' (exactly 1, i.e. classical JSSP), otherwise "
                         "[op_per_mch_min, op_per_mch_max]")
parser.add_argument('--op_per_job', type=float, default=0,
                    help='Number of operations per job, default 0, means the number equals m')
parser.add_argument('--op_per_mch_min', type=int, default=1,
                    help='Minimum number of compatible machines for each operation')
parser.add_argument('--op_per_mch_max', type=int, default=5,
                    help='Maximum number of compatible machines for each operation')
parser.add_argument('--low', type=int, default=1, help='Lower Bound of processing time(PT)')
parser.add_argument('--high', type=int, default=99, help='Upper Bound of processing time')
parser.add_argument('--data_size', type=int, default=100, help='The number of instances for data generation')
parser.add_argument('--cover_data_flag', type=str2bool, default=False, help='Whether covering the generated data')

# args for seed
parser.add_argument('--seed_datagen', type=int, default=1234, help='Seed for data generation')
parser.add_argument('--seed_train', type=int, default=1234, help='Seed for training')
parser.add_argument('--seed_test', type=int, default=1235, help='Seed for testing')

# args for the problem instance
parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of the instance')
parser.add_argument('--n_m', type=int, default=5, help='Number of machines of the instance')

# args for the problem variant (which constraints are active)
parser.add_argument('--flag_ready', type=str2bool, default=False, help='Release dates (R)')
parser.add_argument('--flag_setup', type=str2bool, default=False, help='Sequence-dependent setup time (S)')
parser.add_argument('--flag_maintenance', type=str2bool, default=False, help='Machine maintenance (M)')
parser.add_argument('--flag_trans', type=str2bool, default=False, help='Transportation time (T)')
parser.add_argument('--flag_multi', type=str2bool, default=False,
                    help='Multi-task learning (sample a variant per update) or single-problem learning')
parser.add_argument('--flag_full', type=str2bool, default=False,
                    help='Multi-task learning over all 16 variants (True) or one constraint at a time (False)')

# args for network
parser.add_argument('--num_layer', type=int, default=2, help='Number of main layer')
parser.add_argument('--fea_j_input_dim', type=int, default=2, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=4, help='Dimension of machine raw feature vectors')
parser.add_argument('--fea_e_input_dim', type=int, default=3, help='Dimension of op-machine pair raw feature vectors')
parser.add_argument('--multi_head', type=int, default=8, help='Number of heads')
parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=256, help='Hidden dimension of Actor network')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of Blocks')

# args for model training
parser.add_argument('--run_name', type=str, default='SF_run',
                    help='Name of the training run (results go to ./train_results/{run_name}/FJSSP)')
parser.add_argument('--num_envs', type=int, default=50, help='Number of instances per update (batch size)')
parser.add_argument('--multi_sample', type=int, default=128, help='Number of rollouts sampled per instance')
parser.add_argument('--max_updates', type=int, default=4000, help='Number of training updates')
parser.add_argument('--lr', type=float, default=3 * 1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
parser.add_argument('--model_store_timestep', type=int, default=50, help='Checkpoint saving interval')
parser.add_argument('--accmulation_step', type=int, default=1, help='Gradient accumulation steps')

# args for model inference
parser.add_argument('--test_problem', type=str, default='FJSSP-RSMT',
                    help="Problem variant for testing, e.g. 'FJSSP', 'FJSSP-R', 'FJSSP-RSMT'")
parser.add_argument('--test_size', type=str, default='10x5', help='Instance size for testing, e.g. 10x5')
parser.add_argument('--test_mode', type=str2bool, default=False, help='Whether using the sampling strategy in testing')
parser.add_argument('--sample_times', type=int, default=100, help='Sampling times for the sampling strategy')
parser.add_argument('--test_model', type=str, default='SF_4V', help='Trained model used for testing')
parser.add_argument('--epoch', type=int, default=4000, help='Checkpoint for testing')

# args for heuristic baselines
parser.add_argument('--test_method', nargs='+', default=['FIFO', 'MOR', 'MWKR'],
                    help='Job-selection rules to evaluate (FIFO / MOR / MWKR)')
parser.add_argument('--mch_method', nargs='+', default=['EFT'],
                    help='Machine-selection rules to evaluate (EFT / EST)')
parser.add_argument('--cover_heu_flag', type=str2bool, default=True,
                    help='Whether covering test results of heuristics')

configs = parser.parse_args()
