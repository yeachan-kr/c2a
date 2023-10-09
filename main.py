import os
import sys
import time
import argparse
import datetime
# Custom Library
from utils import set_random_seed, partition_multi_lang_data, Logger


def run(args):

    print(args)
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)

    # Data partitioning based on non-iid strategy
    print('* Partitioning data (num_party: {} by {}, beta: {})'.format(args.n_parties, args.partition, args.beta))
    client2data, _ = partition_multi_lang_data(dataset=args.dataset,
                                                    model=args.model,
                                                    datadir=args.datadir,
                                                    partition=args.partition,
                                                    n_parties=args.n_parties,
                                                    beta=args.beta)

    # Select Solver based on learning strategy
    solver = None
    if args.alg == 'full':
        from solvers.full_fed_avg import FullFedAvgSolver
        solver = FullFedAvgSolver(args=args, client2data=client2data)
    
    if args.alg == 'c2a': 
        from solvers.hyper_fed_avg import HyperFedAvgSolver
        solver = HyperFedAvgSolver(args=args, client2data=client2data)    
    solver.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='multi_sent', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid', help='noniid/iid the data partitioning strategy')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta for the dirichlet distribution for data partitioning')

    # Federated Learning configuration
    parser.add_argument('--sample_fraction', type=float, default=0.25, help='how many clients are sampled in each round')
    parser.add_argument('--n_parties', type=int, default=100, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=20, help='number of maximum communication round')

    # Training configuration
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')

    # Model configuration
    parser.add_argument('--alg', type=str, default='full', help='communication strategy: full/last')
    parser.add_argument('--model', type=str, default='distilbert-base-multilingual-cased', help='backbone pre-trained langauge models')
    parser.add_argument('--rank', type=int, default=16, help='dimension of bottleneck layers')
    parser.add_argument('--con_dim', type=int, default=32, help='dimension of latent factors')
    parser.add_argument('--adapter', type=str, default='hyper', help='dimension of latent factors')

    # Directory configuration conda activate torch37
    parser.add_argument('--datadir', type=str, required=False, default="./data", help="Data directory")
    parser.add_argument('--logdir', type=str, default='./log/', help='dataset used for training')
    parser.add_argument('--modeldir', type=str, default='./save/', help='dataset used for training')

    # Computation configuration
    parser.add_argument('--device', type=str, default='0', help='The device to run the program')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    set_random_seed(args.init_seed)

    # Start solver
    run(args)

