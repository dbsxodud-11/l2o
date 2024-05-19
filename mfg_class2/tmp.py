import csv, os


def load_data(instance_i):
    """ csv file -> python
    """
    setup_time_file = './data/setup_time.csv'

    job_order_file = './data/job_order/job_orders_{}.csv'.format(instance_i)
    wips_file = './data/WIP/work_in_process_{}.csv'.format(instance_i)

    # job_type ops ##################################################
    setup_times = dict()
    with open(setup_time_file, 'r') as f:
        rdr = csv.reader(f)
        for line in rdr:
            line_list = [int(i) for i in line]
            job_type_i = line_list[0]
            setup_times[job_type_i] = line_list[1:]

    # WIP: work in process ##################################################
    wips = list()
    with open(wips_file, 'r') as f:
        rdr = csv.reader(f)
        for job_i, line in enumerate(rdr):
            if job_i == 0:
                stage_mc_n = int(line[0])
                continue

            wips.append([int(line[0]), int(line[1]), int(line[2]), int(line[3])])  # job_type, mc_i, remain_t, due

    # job order ##################################################
    job_order = list()
    if os.path.isfile(job_order_file):
        with open(job_order_file, 'r') as f:
            rdr = csv.reader(f)
            for job_i, line in enumerate(rdr):
                job_order.append([int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])])  # job_type, ready_t, remain_t, due, arrive_t

    return stage_mc_n, setup_times, wips, job_order


#%%


stage_mc_n, setup_times, wips, job_order = load_data(instance_i=0)



#%%

class Job():
    def __init__(self, job_type, prt, ready=0, due=0, id=0):
        self.id = id
        self.type = job_type
        self.prt = prt

        self.ready = ready
        self.due = due

        # dyn info ##########
        self.now_mc_i = -1  # -1: buffer
        self.now_remain_t = 0
        self.buffer_inset_t = -1  # not in buffer
        self.history = list()


import copy



# Visualization: Gantt chart

#%%

import pandas as pd
import plotly.express as px

def get_gantt_plotly(env):
    """
    get gantt chart (web)
    """
    first_list = [(0, 0, '', 0, 0, 0, '')]
    col_name = ['job_type', 'job_id', 'resource', 'start_t', 'end_t', 'due', 'text']
    df = pd.DataFrame(first_list, columns=col_name)

    for mc_i, mc_info in env.mcs.items():
        for his in mc_info['history']:
            job_type, job_id, start_t, end_t, due = his[0], his[1], his[2], his[3], his[4]
            df2 = pd.DataFrame([(job_type, job_id, f'mc_{mc_i}', start_t, end_t, due, str(job_id))], columns=col_name)
            df = pd.concat([df, df2])
    df = df[1:]
    df['delta'] = df['end_t'] - df['start_t']

    fig = px.timeline(df, x_start='start_t', x_end='end_t', y='resource',
                      color='job_type', text='text', opacity=0.7,
                      color_continuous_scale='rainbow')  # https://plotly.com/python/colorscales/
    # px.colors.diverging.BrBG
    # px.colors.sequential.Cividis_r

    fig.update_yaxes(autorange='reversed')

    fig.layout.xaxis.type = 'linear'
    fig.data[0].x = df.delta.tolist()
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(linecolor='black', gridcolor='gray', mirror=True)
    fig.update_yaxes(linecolor='black', mirror=True)
    fig.show()



class JobShopSim():
    def __init__(self, wip_TF=True, setup_type='sequence', instance_i=0, job_arrival_TF=True):
        self.wip_TF = wip_TF
        self.setup_type = setup_type
        self.job_arrival_TF = job_arrival_TF

        # load data #####################################################################
        self.mc_n, self.setup_times, wips, job_order = load_data(instance_i=instance_i)

        # initial #######################################################################
        self.jobs, self.arrival_jobs, self.mcs = dict(), dict(), dict()
        self.sim_t = 0
        self.last_job_idx = 0

        mc_dyn_info = {'doing_job': -1,  # -1: empty
                       'remain_prt': 0,
                       'history': list()
                       }
        for mc_i in range(self.mc_n):
            self.mcs[mc_i] = copy.deepcopy(mc_dyn_info)

        if wip_TF:
            self.init_wips(wips)

        for job_info in job_order:  # job_type, ready_t, prt, due, arrive_t
            arrive_t = job_info[4]
            job = Job(job_type=job_info[0], prt=job_info[2], ready=job_info[1], due=job_info[3], id=self.last_job_idx)
            self.last_job_idx += 1

            if arrive_t not in self.arrival_jobs:
                self.arrival_jobs[arrive_t] = list()
            self.arrival_jobs[arrive_t].append(job)

        self.arrive_jobs()

        # save for reset #################################################################
        self.jobs_init = copy.deepcopy(self.jobs)
        self.mcs_init = copy.deepcopy(self.mcs)
        self.arrival_jobs_init = copy.deepcopy(self.arrival_jobs)

    # initial #############################################################################################
    def init_wips(self, wips):
        """
        create initial stages
        """
        for wip_info in wips:  # # job_type, mc_i, remain_t, due
            mc_i = wip_info[1]

            if mc_i != -1:
                self.mcs[mc_i]['doing_job'] = self.last_job_idx
                self.mcs[mc_i]['remain_prt'] = wip_info[2]
                self.mcs[mc_i]['history'].append((wip_info[0], self.last_job_idx, 0, wip_info[2], wip_info[3]))  # type, id, s_t, e_t, due
                self.last_job_idx += 1

            else:
                self.jobs[self.last_job_idx] = Job(job_type=wip_info[0], prt=wip_info[2], ready=0, due=wip_info[3],
                                                   id=self.last_job_idx)
                self.last_job_idx += 1

    # available ##########################################################################################
    def get_avail_mc_i(self):
        """
        an available machine
        """
        for mc_i, mc_info in self.mcs.items():
            if mc_info['doing_job'] == -1:
                return mc_i
        return -1

    def get_avail_mcs(self):
        """
        an available machine
        """
        avail_mcs = list()
        for mc_i, mc_info in self.mcs.items():
            if mc_info['doing_job'] == -1:
                avail_mcs.append(mc_i)
        return avail_mcs

    def get_mc_last_job_type(self, mc_i):
        """
        for setup of a machine
        """
        return self.mcs[mc_i]['history'][-1][0]

    # job arrival #########################################################################################
    def arrive_jobs(self):
        """
        arrive new jobs
        criteria: self.sim_t <- current simulation time
        """
        del_list = list()
        for arrive_t, jobs in self.arrival_jobs.items():
            if arrive_t <= self.sim_t:
                for job in jobs:
                    self.jobs[job.id] = job
                del_list.append(arrive_t)

        for t in del_list:
            del self.arrival_jobs[t]

    # move state ##########################################################################################
    def move_to_next_sim_t(self):
        """
        1. update job state: end job -> next buffer (or delete job)
        2. update mc state
        3. job arrival
        """
        diff_t = self.get_next_move_t()
        self.sim_t += diff_t

        for mc_i, mc_info in self.mcs.items():
            if mc_info['doing_job'] >= 0:  # doing
                mc_info['remain_prt'] -= diff_t
                if mc_info['remain_prt'] == 0:  # end process
                    mc_info['doing_job'] = -1

        if self.job_arrival_TF:
            self.arrive_jobs()

    def get_next_move_t(self):
        """
        return: minimal move time
        """
        job_remain_ts = [mc_info['remain_prt'] for mc_info in self.mcs.values() if mc_info['remain_prt'] > 0]

        if self.job_arrival_TF and self.arrival_jobs:
            job_remain_ts += [min(self.arrival_jobs.keys()) - self.sim_t]

        if not job_remain_ts:
            return 0
        return min(job_remain_ts)

    def start_job(self, job_i, mc_i):
        """
        1. pop the job in the buffer
        2. update job state
        3. update mc state

        when
        job selection in the buffer
        """
        job = self.jobs[job_i]
        mc_last_job_type = self.get_mc_last_job_type(mc_i)
        if 'constant' in self.setup_type:
            if job.type == mc_last_job_type:
                setup_t = 0
            else:
                setup_t = self.setup_times[0][1]
        elif 'sequence' in self.setup_type:
            setup_t = self.setup_times[mc_last_job_type][job.type]
        else:
            setup_t = 0

        s_t = max(job.ready, self.sim_t + setup_t)

        self.mcs[mc_i]['doing_job'] = job.id
        self.mcs[mc_i]['remain_prt'] = s_t + job.prt - self.sim_t

        if setup_t > 0:
            self.mcs[mc_i]['history'].append((-1, -1, self.sim_t, self.sim_t + setup_t, 0))
        self.mcs[mc_i]['history'].append((job.type, job.id, s_t, s_t + job.prt, job.due))

        del self.jobs[job.id]
        del job


i = 11
env = JobShopSim(wip_TF=True, setup_type='constant', instance_i=i)

while env.jobs:
    while env.jobs and env.get_avail_mc_i() >= 0:
        mc_i = env.get_avail_mc_i()

        # EDD rule
        jobs = list(env.jobs.values())
        jobs.sort(key=lambda x: x.due)
        job_i = jobs[0].id

        env.start_job(job_i, mc_i)

    env.move_to_next_sim_t()

# for mc_i, mc_info in env.mcs.items():
#     print(mc_i, mc_info['history'])

# get_gantt_plotly(env)


#%% md

# Environment for JSSP: interaction with dispatching agent

#%%

class JobShopEnv(JobShopSim):
    def __init__(self, wip_TF=False, setup_type='constant', obj='tardiness', instance_i=0, job_arrival_TF=True):
        super().__init__(wip_TF=wip_TF, setup_type=setup_type, instance_i=instance_i, job_arrival_TF=job_arrival_TF)
        self.obj = obj

    def step(self, job_i, mc_i):
        """
        action -> next state
        """
        self.start_job(job_i, mc_i)

        return self.move_next_state()

    def move_next_state(self):
        """
        check all stages
        if there is one job in the buffer -> automatic assign to the first machine
        else -> decision point
        """
        assign_job_ids = list()

        while not self.is_done():
            while self.jobs and self.get_avail_mc_i() >= 0:
                mc_i = self.get_avail_mc_i()
                if len(list(self.jobs.keys())) == 1:  # automatic assign
                    job_i = list(self.jobs.keys())[0]
                    self.start_job(job_i, mc_i)
                    assign_job_ids.append((job_i, mc_i))
                    continue

                return mc_i, self.jobs, False, assign_job_ids

            self.move_to_next_sim_t()

        return -1, None, self.is_done(), assign_job_ids

    def get_obj(self):
        if self.obj == 'tardiness':
            return self.get_total_tardiness(), self.get_setups()
        elif self.obj == 'makespan':
            return self.sim_t, self.get_setups()
        else:
            raise NotImplementedError

    def get_total_tardiness(self):
        """
        return  total tardiness
        """
        tardiness = 0
        for mc_info in self.mcs.values():
            for assign_job in mc_info['history']:
                if assign_job[0] == -1:
                    continue
                tardiness += max(0, assign_job[3] - assign_job[4])
                # print(max(0, assign_job[3] - assign_job[4]))
        return tardiness

    def get_setups(self):
        """
        return  total tardiness
        """
        setups = 0
        for mc_info in self.mcs.values():
            for assign_job in mc_info['history']:
                if assign_job[0] == -1:
                    setups += assign_job[3] - assign_job[2]
        return setups

    def is_done(self) -> bool:
        """
        return True if the simulation is done
        """
        if self.job_arrival_TF:
            if not self.jobs and not self.arrival_jobs:
                if sum([mc_info['remain_prt'] for mc_info in self.mcs.values()]) <= 0:
                    return True
            return False
        else:
            if not self.jobs:
                if sum([mc_info['remain_prt'] for mc_info in self.mcs.values()]) <= 0:
                    return True
            return False

    def reset(self):
        """
        reset environment
        """
        self.jobs = copy.deepcopy(self.jobs_init)
        self.mcs = copy.deepcopy(self.mcs_init)
        self.arrival_jobs = copy.deepcopy(self.arrival_jobs_init)

        self.sim_t = 0
        self.last_job_idx = 0

        return self.move_next_state()


#%%

obj = 'tardiness'
instance_i = 11  # index 10: simple test
env = JobShopEnv(wip_TF=True, obj=obj, setup_type='sequence', instance_i=instance_i)

mc_i, job_dict, done, _ = env.move_next_state()
while not done:
    # EDD rule
    jobs = list(env.jobs.values())
    jobs.sort(key=lambda x: x.due)
    job_i = jobs[0].id

    mc_i, job_dict, done, _ = env.step(job_i, mc_i)

print(f'instance {i} - obj {env.get_obj()} - makespan: {env.sim_t}')
# get_gantt_plotly(env)

#%% md

# Dispatching agent

#%%


def run_episode_envs(obj='tardiness', instance_is=[0], save_path='',
                     method='rule', rule='EDD', decision_tree=None, chromosome=None):
    """
    generate a schedule by using a dispatching rule
    diff_stage_TF == True: each stage has different decision tree
    """
    envs = list()
    for instance_i in instance_is:
        env = JobShopEnv(wip_TF=True, obj=obj, setup_type='constant', instance_i=instance_i)
        envs.append(env)

    sum_obj = 0
    for i, env in enumerate(envs):
        obj, setup = run_episode(env, method=method, rule=rule, decision_tree=decision_tree, chromosome=chromosome)
        sum_obj += obj

        if save_path:
            with open(save_path, 'a', newline='') as f:
                wr = csv.writer(f)
                wr.writerow([method, rule, instance_is[i], obj, setup])

    return round(sum_obj / len(instance_is))


def run_episode(env, method='rule', rule='EDD', decision_tree=None, chromosome=None, reset=False):
    """
    generate a schedule by using a dispatching rule
    diff_stage_TF == True: each stage has different decision tree
    """
    if reset:
        mc_i, job_dict, done, assign_job_ids = env.reset()
    else:
        mc_i, job_dict, done, assign_job_ids = env.move_next_state()

    if method == 'chrm':
        copy_chromosome = copy.deepcopy(chromosome)
        copy_chromosome = update_chromosome(copy_chromosome, assign_job_ids)

    while not done:
        factory_info = get_factory_info(env, mc_i)
        if method == 'rule':
            job_i, mc_i = get_action_rule(job_dict, mc_i, rule=rule, factory_info=factory_info)
        elif method == 'chrm':
            job_i, mc_i = get_action_chrm(job_dict, mc_i, factory_info=factory_info, chromosome=copy_chromosome)
        elif method == 'tree':
            job_i, mc_i = get_action_tree(job_dict, mc_i, factory_info=factory_info, model=decision_tree)
        else:
            raise NotImplementedError

        mc_i, job_dict, done, assign_job_ids = env.step(job_i, mc_i)

        if method == 'chrm':
            assign_job_ids.append((job_i, mc_i))
            copy_chromosome = update_chromosome(copy_chromosome, assign_job_ids)
            if not copy_chromosome:
                done = True

    return env.get_obj()


import random
def get_action_rule(job_dict, mc_i, rule, factory_info=None):
    """
    return job_index having the largest value computed by the rule
    """
    jobs = list(job_dict.values())

    if 'SPT' in rule:  # shortest processing time
        jobs.sort(key=lambda x: x.prt)
    elif 'LPT' in rule:  # longest processing time
        jobs.sort(key=lambda x: -x.prt)
    elif 'FIFO' in rule:  # first-in-first-out
        jobs.sort(key=lambda x: x.ready)
    elif 'LIFO' in rule:  # last-in-first-out
        jobs.sort(key=lambda x: -x.ready)
    elif 'EDD' in rule:  # earliest due date
        jobs.sort(key=lambda x: x.due)
    elif rule == 'random':
        job_i = random.choice(list(job_dict.keys()))
        return job_i, mc_i
    ##########################################################################################################
    # TODO: allow edit for new dispatching rules
    elif 'LDD' in rule:  # latest due date
        jobs.sort(key=lambda x: x.due)
    else:
        raise NotImplementedError
    ##########################################################################################################

    return jobs[0].id, mc_i


def get_factory_info(env, mc_i):
    """
    calculate load for each stage
    """
    factory_info = dict()
    ##########################################################################################################
    # TODO: allow edit for learning decision tree

    factory_info['mc_last_job_type'] = env.get_mc_last_job_type(mc_i)

    ##########################################################################################################

    return factory_info


mean_obj = run_episode_envs(save_path='result_rule.csv', method='rule', rule='random', instance_is=list(range(24)))
# mean_obj = run_episode_envs(save_path='result_rule.csv', method='rule', rule='random', instance_is=list(range(24)))
print(mean_obj)






# TODO: allow edit for chromosome-based evolutionary methods #################


def get_action_chrm(job_dict, mc_i, factory_info=None, chromosome=[]):
    """
    return a job selected by the chromosome

    job_dict -> select the job appeared the earliest in the chromosome
    chromosome: list(str: job_id)
    """
    jobs = list(job_dict.values())

    earliest_pos = len(chromosome)
    select_job = None
    for job in jobs:
        job_pos = chromosome.index(job.id)

        if earliest_pos > job_pos:
            earliest_pos = job_pos
            select_job = job

    return select_job.id, mc_i


def update_chromosome(chromosome, assign_job_ids):
    for job_i, mc_i in assign_job_ids:
        chromosome.remove(job_i)
    return chromosome




import random
import numpy as np


def get_genes(env):
    chrm = list()
    for job in env.jobs.values():
        chrm.append(job.id)

    return chrm


def get_chrm_from_rule(env, rule):
    env_ = copy.deepcopy(env)

    # update history for dynamic environment
    for mc_i, mc_info in env_.mcs.items():
        if env_.wip_TF:
            mc_info['history'] = mc_info['history'][-1:]
        else:
            mc_info['history'].clear()
    run_episode(env_, method='rule', rule=rule, reset=False)

    chrm = []
    for mc_i, mc_info in env_.mcs.items():
        for i, his in enumerate(mc_info['history']):
            if env.wip_TF and i == 0:
                continue

            job_id = his[1]
            if job_id != -1:
                start_t = his[2]
                if i > 0 and mc_info['history'][i-1][1] == -1:
                    start_t = mc_info['history'][i-1][2]  # setup start time
                chrm.append((job_id, start_t))

    chrm.sort(key=lambda x: x[1])
    chrm = [gene[0] for gene in chrm]
    return chrm


def get_init_population(env, job_list, pop_size):
    rules = ['SPT', 'LPT', 'FIFO', 'LIFO', 'EDD']
    rules = rules[:pop_size]
    init_pop = [get_chrm_from_rule(env, rule) for rule in rules]
    init_pop = init_pop + [random.sample(job_list, len(job_list))
                           for _ in range(pop_size - len(rules))]
    return init_pop


def get_fitness(env, pop_):
    obj_list = []
    for chrm in pop_:
        env_ = copy.deepcopy(env)
        obj_list.append(run_episode(env_, method='chrm', chromosome=chrm)[0])
    fit_list = obj_list
    max_fit = max(fit_list)
    fit_list = [(max_fit - fit) for fit in fit_list]
    return fit_list, obj_list


def get_best(pop_, fit_list, obj_list):
    best_idx = np.argmax(fit_list)
    return copy.copy(pop_[best_idx]), obj_list[best_idx]


def print_best_obj(best_obj, gen):
    if (gen%10)==0:
        print(f'gen=={gen}, best_obj=={best_obj}')


def terminate(gen, gen_num):
    terminate_TF = (gen==gen_num)
    return terminate_TF


def selection(pop_, fit_list, pop_size):
    sum_fit = sum(fit_list)
    prob_list = [(fit/sum_fit) if sum_fit else (1/len(fit_list)) for fit in fit_list]
    idx_list = list(range(pop_size))
    pairs = [(pop_[np.random.choice(idx_list, p=prob_list)],
              pop_[np.random.choice(idx_list, p=prob_list)])
             for _ in range(pop_size)]
    return pairs


def two_point_cross(pair):
    chrm1 = []
    chrm2 = []
    count_ = {job_id:0 for job_id in list(set(pair[0]))}
    for job_id in pair[0]:
        chrm1.append((job_id, count_[job_id]))
        count_[job_id] += 1
    count_ = {k:0 for k in count_}
    for job_id in pair[1]:
        chrm2.append((job_id, count_[job_id]))
        count_[job_id] += 1
    oper_to_idx = {chrm1[i]:i for i in range(len(chrm1))}
    idx_to_oper = {oper_to_idx[oper]:oper for oper in oper_to_idx}

    chrm1 = [oper_to_idx[oper] for oper in chrm1]
    chrm2 = [oper_to_idx[oper] for oper in chrm2]
    points = sorted(random.sample(list(range(0,len(chrm1)+1)),2))
    p1 = points[0]
    p2 = points[1]
    inter = chrm1[p1:p2]
    left = [e for e in chrm2 if e not in inter]
    child = left[:p1] + inter + left[len(left)-(len(chrm1)-p2):]
    child = [idx_to_oper[idx][0] for idx in child]
    return child


def crossover(pairs, cross_prob):
    pop_ = [two_point_cross(pair)
            if (random.random() < cross_prob)
            else pair[random.randint(0,1)]
            for pair in pairs]
    return pop_


def mutation(pop_, mut_prob):
    pop_ = [random.sample(chrm, len(chrm)) if (random.random() < mut_prob)
            else chrm
            for chrm in pop_]
    return pop_


def local_search(env,  pop_, hyb_prob, hyb_iter):
    idx = list(range(len(pop_[0])))
    for i in range(len(pop_)):
        if random.random() < hyb_prob:
            chrm = pop_[i]
            copy_env = copy.deepcopy(env)
            best_obj = run_episode(copy_env, method='chrm', chromosome=chrm)
            n = random.randrange(len(idx))
            idx1 = random.sample(idx, n)
            for _ in range(hyb_iter):
                idx2 = random.sample(idx1, len(idx1))
                new_chrm = copy.copy(chrm)
                for k in range(len(idx1)):
                    new_chrm[idx1[k]] = chrm[idx2[k]]
                copy_env = copy.deepcopy(env)
                new_obj = run_episode(copy_env, method='chrm', chromosome=new_chrm)
                if new_obj < best_obj:
                    pop_[i] = new_chrm
                    best_obj = new_obj
    return pop_


def insert_best(pop_, best):
    pop_.pop(random.randrange(len(pop_)))
    pop_.append(best)
    return pop_






def run_GA(env, configs):
    pop_size, cross_prob, mut_prob, max_gen, hyb_prob, hyb_iter = \
        configs['population_size'], configs['crossover_prob'], configs['mutation_prob'], \
        configs['maximum_generation'], configs['hybrid_prob'], configs['hybrid_iteration']

    genes = get_genes(env)
    gen = 0
    pop_ = get_init_population(env, genes, pop_size)
    while True:
        fit_list, obj_list = get_fitness(env, pop_)
        best_chrm, best_obj = get_best(pop_, fit_list, obj_list)
        print_best_obj(best_obj, gen)
        if terminate(gen, max_gen):
            break
        pairs = selection(pop_, fit_list, pop_size)
        pop_ = crossover(pairs, cross_prob)
        pop_ = mutation(pop_, mut_prob)
        pop_ = local_search(env, pop_, hyb_prob, hyb_iter)
        pop_ = insert_best(pop_, best_chrm)
        gen += 1

    return best_chrm

def new_job_arrived(env, prev_job_ids):
    new_job_TF = False
    for job_id in env.jobs.keys():
        if job_id not in prev_job_ids:
            new_job_TF = True
            print(f'new_job_TF ---------- {new_job_TF}')
            break
    return new_job_TF

def fix_chromosome(best_chromosome, job_id, assign_job_ids):
    best_chromosome.remove(job_id)
    for job_id in assign_job_ids:
        best_chromosome.remove(job_id)
    return best_chromosome

def run_dynamic_GA(env, configs):
    mc_i, job_dict, done, _ = env.reset()
    prev_job_ids = []

    while not done:
        if new_job_arrived(env, prev_job_ids):
            copy_env = copy.deepcopy(env)
            copy_env.job_arrival_TF = False
            best_chromosome = run_GA(copy_env, configs)

        prev_job_ids = list(env.jobs.keys())
        factory_info = get_factory_info(env, mc_i)
        job_i, mc_i = get_action_chrm(job_dict, mc_i, factory_info=factory_info, chromosome=best_chromosome)
        mc_i, job_dict, done, assign_job_ids = env.step(job_i, mc_i)
        assign_job_ids.append((job_i, mc_i))
        best_chromosome = update_chromosome(best_chromosome, assign_job_ids)

    return env.get_obj()




obj = 'tardiness'

i = 1
env = JobShopEnv(wip_TF=True, obj=obj, setup_type='sequence', instance_i=i)

"""Parameters of GA"""
configs = {'population_size': 10,
           'crossover_prob': 0.8,
           'mutation_prob': 0.1,
           'maximum_generation': 20,
           'hybrid_prob': 0.5,
           'hybrid_iteration': 5,
           }

obj_value = run_dynamic_GA(env, configs)
print("performance of GA for instance {}: {} - makespan: {}".format(i, obj_value, env.sim_t))

get_gantt_plotly(env)





# TODO: allow edit for learning decision tree ###############################

def get_action_tree(job_dict, mc_i, factory_info=None, model=None):
    """
    return a job selected by the chromosome

    job_dict -> select the job appeared the earliest in the chromosome
    chromosome: list(str: job_id)
    """
    jobs = list(job_dict.values())

    def sort_by_model(partial_jobs, model, factory_info):
        job_n = len(partial_jobs)
        searched_info = dict()
        searched_info[partial_jobs[0].id] = [partial_jobs[1].id]

        while True:
            job1 = partial_jobs[0]
            insert_pos = 0
            for compare_pos in range(1, job_n):
                job2 = partial_jobs[compare_pos]
                data = get_DT_data(job1, job2, factory_info)
                job1_first_TF = model.predict([data])[0]
                if not job1_first_TF:
                    insert_pos = compare_pos
                else:
                    break

            # insert job
            if insert_pos >= 1:
                searched_info[partial_jobs[0].id] = [partial_jobs[1].id]
                # if insert_pos < job_n - 1:
                #     searched_info[partial_jobs[0].id] = [partial_jobs[insert_pos].id]
                partial_jobs.insert(insert_pos, partial_jobs.pop(0))
            else:
                break

            # avoid unlimited run
            if partial_jobs[0].id in searched_info.keys():
                if partial_jobs[1].id in searched_info[partial_jobs[0].id]:
                    break

        return partial_jobs

    for i in range(2, len(jobs)+1):
        partial_jobs = jobs[-i:]
        jobs = jobs[:-i]
        jobs += sort_by_model(partial_jobs, model, factory_info)

    return jobs[0].id, mc_i


def get_DT_data(job1, job2, factory_info):
    """
    get input data for a decision tree model
    factory_info <- last_job_type, stage_n_load, stage_prt_load, stage_buffer_n
    """
    ##########################################################################################################
    # TODO: allow edit for learning decision tree

    last_job_type = factory_info['mc_last_job_type']

    data = (job1.type == last_job_type, job2.type == last_job_type,
        job1.due > job2.due, job1.due == job2.due,
        job1.prt > job2.prt, job1.prt == job2.prt,
        )

    ##########################################################################################################

    return data


data_col_name = ['job1_setup_TF', 'job2_setup_TF',
                 'due_job1_large', 'due_same',
                 'prt_job1_large', 'prt_same',
                 ]


def run_episode_rule_save_decisions(env, method='', rule='', configs=None):
    """
    generate a schedule by using a dispatching rule
    diff_stage_TF == True: you will learn a decision tree for each stage
    """
    decisions = list()
    targets = list()

    mc_i, job_dict, done, _ = env.reset()

    if method == 'chrm':
        copy_env = copy.deepcopy(env)
        copy_env.job_arrival_TF = False
        if rule == 'GA':
            best_chromosome = run_GA(copy_env, configs)
        else:
            raise NotImplementedError
        prev_job_ids = list(env.jobs.keys())

    else:
        best_chromosome = None

    while not done:
        factory_info = get_factory_info(env, mc_i)
        if method == 'rule':
            job_i, mc_i = get_action_rule(job_dict, mc_i, rule=rule, factory_info=factory_info)
        elif method == 'chrm':
            job_i, mc_i = get_action_chrm(job_dict, mc_i, factory_info=factory_info, chromosome=best_chromosome)
        else:
            raise NotImplementedError

        if len(env.mcs[mc_i]['history']) > 1:
            for other_job in job_dict.values():
                if other_job.id == job_i:
                    continue
                job = job_dict[job_i]
                targets.append(True)
                decisions.append(get_DT_data(job, other_job, factory_info))
                targets.append(False)
                decisions.append(get_DT_data(other_job, job, factory_info))

        mc_i, job_dict, done, assign_job_ids = env.step(job_i, mc_i)

        if method == 'chrm':
            assign_job_ids.append((job_i, mc_i))
            best_chromosome = update_chromosome(best_chromosome, assign_job_ids)

            if new_job_arrived(env, prev_job_ids):
                copy_env = copy.deepcopy(env)
                copy_env.job_arrival_TF = False
                if rule == 'GA':
                    best_chromosome = run_GA(copy_env, configs)
                else:
                    raise NotImplementedError

            prev_job_ids = list(env.jobs.keys())

    return env.get_obj(), decisions, targets


def run_rules_save_decisions(env, methods, save_ratio=0.1, configs=None):
    """
    generate dataset for decision tree
    diff_stage_TF == True: you will learn a decision tree for each stage
    """

    dataset = list()
    for method, rule in methods:
        obj, decisions, targets = run_episode_rule_save_decisions(env, method=method, rule=rule, configs=configs)
        dataset.append((obj, method, rule, decisions, targets))
        print(f'method: {method}, rule: {rule} - objective: {obj} - makespan: {env.sim_t}')
        # get_gantt_plotly(env, save_TF=True, rule=rule, instance_i=instance_i)

    ''' select good schedules '''
    dataset.sort(key=lambda x: x[0])  # sort by obj
    select_n = max(int(len(methods) * save_ratio), 1)  # save top 10% schedules
    dataset = dataset[:select_n]
    selected_rules = list()

    save_decisions = list()
    save_targets = list()
    for _, method, rule, decisions, targets in dataset:
        selected_rules.append((method, rule))
        save_decisions += decisions
        save_targets += targets

    print('selected rules ', selected_rules)
    return save_decisions, save_targets








# Construct decision tree

methods = [('chrm', 'GA') for _ in range(1)]
# [('rule', 'SPT'), ('rule', 'LPT'), ('rule', 'FIFO'), ('rule', 'LIFO'), ('rule', 'EDD'), ] + \
#  \
"""Parameters of GA"""
configs = {'population_size': 50,
           'crossover_prob': 0.8,
           'mutation_prob': 0.1,
           'maximum_generation': 1,
           'hybrid_prob': 0.5,
           'hybrid_iteration': 2,
           }

obj = 'tardiness'
save_ratio = 0.01

decisions = list()
targets = list()
for instance_i in range(1):
    env = JobShopEnv(wip_TF=True, obj=obj, setup_type='sequence', instance_i=instance_i)
    save_decisions, save_targets = run_rules_save_decisions(env, methods, configs=configs)

    decisions += save_decisions
    targets += save_targets

# data num
print("# of data: {} {}".format(len(decisions), len(targets)))









from sklearn.tree import DecisionTreeClassifier

def get_tree_model(max_depth, x, y):
    """
    learning decision tree
    diff_stage_TF == True: you will learn a decision tree for each stage
    """
    tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    tree.fit(x, y)
    # print("train accuracy: {:.3f}".format(tree.score(x, y)))

    return tree




from sklearn.tree import export_graphviz

obj = 'tardiness'

depth_weight = 1  # larger depth --> hard to analysis
best_obj = float('inf')
best_tree = None
for max_depth in range(2, 3):
    print("------------- max depth: {} ---------------".format(max_depth))

    # all data ##################################################################################
    tree = get_tree_model(max_depth, decisions, targets)
    mean_obj = run_episode_envs(obj=obj, method='tree', decision_tree=tree, instance_is=list(range(2)))
    print("training performance: {:.3f}".format(mean_obj))

    mean_obj += max_depth * depth_weight
    if best_obj > mean_obj:
        best_obj = mean_obj
        print("==== save ====")
        best_tree = tree

        # save for tree visualizing
        export_graphviz(tree, out_file='./model/DT_model.dot', feature_names=data_col_name, filled=True)



for instance_i in range(2):
    env = JobShopEnv(wip_TF=True, obj=obj, setup_type='sequence', instance_i=instance_i)
    obj_v = run_episode(env, method='tree', decision_tree=best_tree)
    print(f'instance: {instance_i} - objective: {obj_v} - makespan: {env.sim_t}')



import os
import graphviz
os.environ['PATH'] += os.pathsep + 'D:/Scheduling_Simulator/mfg_class2/Graphviz/bin'

with open('./model/DT_model.dot') as f:
    dot_graph = f.read()
src = graphviz.Source(dot_graph, format='png')
src.view(filename='tree_structure', cleanup=True)




import matplotlib.pyplot as plt

def plot_feature_importance(model, n_features, data_col_name):
    """
    visualize feature importance of the decision tree model
    """
    plt.barh(range(n_features), model.feature_importances_)
    plt.yticks(np.arange(n_features), data_col_name)
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
    plt.show()


plot_feature_importance(best_tree, len(data_col_name), data_col_name)


print()