import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from DataGenerator import TSPDataset
from tqdm import tqdm
from TSPEnvironment import TSPInstanceEnv, VecEnv
from ActorCriticNetwork import ActorCriticNetwork

from schedule_env import *


parser = argparse.ArgumentParser(description='TSPNet')

# ----------------------------------- Data ---------------------------------- #
parser.add_argument('--test_size',
                    default=1, type=int, help='Test data size')
parser.add_argument('--test_from_data',
                    default=True,
                    action='store_true', help='Render')
parser.add_argument('--n_points',
                    type=int, default=20, help='Number of points in TSP')
# ---------------------------------- Train ---------------------------------- #
parser.add_argument('--n_steps',
                    default=20,
                    type=int, help='Number of steps in each episode')
parser.add_argument('--render',
                    default=True,
                    action='store_true', help='Render')
# ----------------------------------- GPU ----------------------------------- #
parser.add_argument('--gpu',
                    default=True, action='store_true', help='Enable gpu')
# --------------------------------- Network --------------------------------- #
parser.add_argument('--input_dim',
                    type=int, default=2, help='Input size')
parser.add_argument('--embedding_dim',
                    type=int, default=128, help='Embedding size')
parser.add_argument('--hidden_dim',
                    type=int, default=128, help='Number of hidden units')
parser.add_argument('--n_rnn_layers',
                    type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--n_actions',
                    type=int, default=2, help='Number of nodes to output')
parser.add_argument('--graph_ref',
                    default=False,
                    action='store_true',
                    help='Use message passing as reference')
parser.add_argument('--i', default=11)

# --------------------------------- Misc --------------------------------- #
parser.add_argument('--load_path', type=str,
    default='best_policy/policy-TSP100-epoch-262.pt')
parser.add_argument('--data_dir', type=str, default='data')

args = parser.parse_args()

if args.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices available.' % torch.cuda.device_count())
else:
    USE_CUDA = False

# loading the model from file
if args.load_path != '':
    print('  [*] Loading model from {}'.format(args.load_path))

    model = ActorCriticNetwork(args.input_dim,
                               args.embedding_dim,
                               args.hidden_dim,
                               args.n_points,
                               args.n_rnn_layers,
                               args.n_actions,
                               args.graph_ref)
    checkpoint = torch.load(os.path.join(os.getcwd(), args.load_path))
    policy = checkpoint['policy']
    model.load_state_dict(policy)

# Move model to the GPU
if USE_CUDA:
    model.cuda()

model = model.eval()


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

obj = 'tardiness'

i = args.i
env = JobShopEnv(wip_TF=True, obj=obj, setup_type='sequence', instance_i=i)

mc_i, job_dict, done, _ = env.reset()
prev_job_ids = []
batch_size = 10

while not done:
    if new_job_arrived(env, prev_job_ids):
        copy_env = copy.deepcopy(env)
        copy_env.job_arrival_TF = False
        # best_chromosome = run_GA(copy_env, configs)
        # best_chromosome = np.random.permutation(list(env.jobs.keys())).tolist()
        # best_chromosome = run_LGA(copy_env, configs)
        env_ = copy.deepcopy(copy_env)
        # chrm = np.random.permutation(list(env_.jobs.keys())).tolist()
        # print(run_episode(env_, method='chrm', chromosome=chrm)[0])
        # print(kyle)
        rules = ['SPT', 'LPT', 'FIFO', 'LIFO', 'EDD']
        init_pop = [get_chrm_from_rule(env_, rule) for rule in rules]
        chromosome = init_pop + [np.random.permutation(list(env_.jobs.keys())).tolist() for _ in range(batch_size - len(init_pop))]
        # chromosome = get_chrm_from_rule(env, 'EDD')
        # chromosome = np.random.permutation(list(env.jobs.keys())).tolist()
        
        state = torch.zeros((batch_size, len(chromosome[0]), 2)).cuda()
        for j, k in enumerate(list(env.jobs.values())):
            state[:, j, 0] = torch.FloatTensor([k.ready for _ in range(batch_size)]).cuda()
            state[:, j, 1] = torch.FloatTensor([k.due for _ in range(batch_size)]).cuda()
        
        obj_list = []
        for b in range(batch_size):
            env_ = copy.deepcopy(copy_env)
            obj = run_episode(env_, method='chrm', chromosome=chromosome[b])[0]
            obj_list.append(obj)
        print(f"step==0\tbest_obj=={min(obj_list)}")
        
        best_chromosome = copy.deepcopy(chromosome) 
        best_state = state.clone()
        # best_obj = obj
        best_obj_list = obj_list
        hidden = None
        
        for step in range(200):
            with torch.no_grad():
                _, action, _, _, _, hidden = model(state, best_state, hidden)
            # print(action.shape)
            new_chromosome = copy.deepcopy(chromosome)
            for b in range(batch_size):
                new_chromosome[b][action[b, 0]] = chromosome[b][action[b, 1]]
                new_chromosome[b][action[b, 1]] = chromosome[b][action[b, 0]]
            
            new_state = state.clone()
            for b in range(batch_size):
                new_state[b, action[b, 0]] = state[b, action[b, 1]]
                new_state[b, action[b, 1]] = state[b, action[b, 0]]
            
            new_obj_list = []
            for b in range(batch_size):
                env_ = copy.deepcopy(copy_env)
                new_obj = run_episode(env_, method='chrm', chromosome=new_chromosome[b])[0]
                new_obj_list.append(new_obj)
            # state = new_state
            # chromosome = new_chromosome
                if new_obj < best_obj_list[b]:
                    state[b] = new_state[b]
                    chromosome[b] = new_chromosome[b]
                    
                    best_chromosome[b] = new_chromosome[b]
                    best_state[b] = new_state[b]
                    # print(best_obj)
                    # best_obj = new_obj
                    best_obj_list[b] = new_obj
            
            # if (step+1) % 10 == 0:
            #     print(f"step=={step+1}\tbest_obj=={min(best_obj_list)}")

        best_chromosome = best_chromosome[np.array(best_obj_list).argmin()]
                
    prev_job_ids = list(env.jobs.keys())
    factory_info = get_factory_info(env, mc_i)
    job_i, mc_i = get_action_chrm(job_dict, mc_i, factory_info=factory_info, chromosome=best_chromosome)
    mc_i, job_dict, done, assign_job_ids = env.step(job_i, mc_i)
    assign_job_ids.append((job_i, mc_i))
    best_chromosome = update_chromosome(best_chromosome, assign_job_ids)

obj_value = env.get_obj()
print("performance of L2O for instance {}: {} - makespan: {}".format(i, obj_value, env.sim_t))
np.savez_compressed(f"results/performance_{i}.npz", tardiness=obj_value[0], setup_time=obj_value[1], makespan=env.sim_t)



# mc_i, job_dict, done, _ = env.reset()
# prev_job_ids = []
# print
# mc_i, job_dict, done, assign_job_ids = env.reset()
# chromosome = np.random.permutation(list(job_dict.keys())).tolist()
# print(chromosome)

# while not done:
#     factory_info = get_factory_info(env, mc_i)
#     print(job_dict)
#     job_i, mc_i = get_action_chrm(job_dict, mc_i, factory_info=factory_info, chromosome=chromosome)
#     mc_i, job_dict, done, assign_job_ids = env.step(job_i, mc_i)
# print(env.get_obj())
# obj = run_episode(env, method="chrm", chromosome=chromosome)
# state = torch.zeros((1, len(chromosome), 2)).cuda()
# id2idx = {}
# for i, j in enumerate(list(env.jobs.values())):
#     id2idx[i] = j.id
#     state[0, i, 0] = j.prt
#     state[0, i, 1] = j.ready
    
# best_state = state.clone()
# hidden = None


# with torch.no_grad():
#     _, action, _, _, _, hidden = model(state, best_state, hidden)
# print(action)
# new_chromosome = chromosome.clone()
# print(new_chromosome)
# new_chromosome[action[0, 0]] = chromosome[action[0, 1]]
# new_chromosome[action[0, 1]] = chromosome[action[0, 0]]
# print(new_chromosome)

# copy_env = copy.deepcopy(env)
# new_obj = run_episode(copy_env, method="chrm", chromosome=new_chromosome)
# print(obj, new_obj)


# performance of GA for instance 11: (80536, 10908) - makespan: 4810
# performance of L2O for instance 11: (66868, 9661) - makespan: 4589
# performance of L2O for instance 11: (67955, 10139) - makespan: 4692
