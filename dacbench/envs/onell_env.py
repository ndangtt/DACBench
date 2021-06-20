from os import fdatasync
import numpy as np
from copy import deepcopy
import logging
from collections import deque

import sys
import os
import uuid
import gym

from dacbench import AbstractEnv

class BinaryProblem:
    """
    An abstract class for an individual in binary representation
    """
    def __init__(self, n, val=None, rng=np.random.default_rng()):
        if val is not None:
            assert isinstance(val, bool)
            self.data = np.array([val] * n)
        else:
            self.data = rng.choice([True,False], size=n) 
        self.n = n
        self.fitness = self.eval()

    
    def initialise_with_fixed_number_of_bits(self, k, rng=np.random.default_rng()):
        nbits = self.data.sum()        
        if nbits < k:            
            ids = rng.choice(np.where(self.data==False)[0], size=k-nbits, replace=False)
            self.data[ids] = True
            self.eval()
        

    def is_optimal(self):
        pass


    def get_optimal(self):
        pass


    def eval(self):
        pass        


    def get_fitness_after_flipping(self, locs):
        """
        Calculate the change in fitness after flipping the bits at positions locs

        Parameters
        -----------
            locs: 1d-array
                positions where bits are flipped

        Returns: int
        -----------
            objective after flipping
        """
        raise NotImplementedError

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """
        Calculate fitness of the child aftering being crossovered with xprime

        Parameters
        -----------
            xprime: 1d boolean array
                the individual to crossover with
            locs_x: 1d boolean/integer array
                positions where we keep current bits of self
            locs_xprime: : 1d boolean/integer array
                positions where we change to xprime's bits

        Returns: fitness of the new individual after crossover
        -----------            
        """
        raise NotImplementedError

    def flip(self, locs):
        """
        flip the bits at position indicated by locs

        Parameters
        -----------
            locs: 1d-array
                positions where bits are flipped

        Returns: the new individual after the flip
        """
        child = deepcopy(self)
        child.data[locs] = ~child.data[locs]
        child.eval()
        return child

    def combine(self, xprime, locs_xprime):
        """
        combine (crossover) self and xprime by taking xprime's bits at locs_xprime and self's bits at other positions

        Parameters
        -----------
            xprime: 1d boolean array
                the individual to crossover with
            locs_x: 1d boolean/integer array
                positions where we keep current bits of self
            locs_xprime: : 1d boolean/integer array
                positions where we change to xprime's bits

        Returns: the new individual after the crossover        

        """
        child = deepcopy(self)
        child.data[locs_xprime] = xprime.data[locs_xprime]
        child.eval()
        return child

    def mutate(self, p, n_childs, rng=np.random.default_rng()):
        """
        Draw l ~ binomial(n, p), l>0
        Generate n_childs children by flipping exactly l bits
        Return: the best child (maximum fitness), its fitness and number of evaluations used        
        """
        assert p>=0

        if p==0:
            return self, self.fitness, 0

        l = 0
        while l==0:
            l = rng.binomial(self.n, p)                
        
        best_obj = -1
        best_locs = None
        for i in range(n_childs):
            locs = rng.choice(self.n, size=l, replace=False)        
            obj = self.get_fitness_after_flipping(locs)
            if obj > best_obj:
                best_locs = locs
                best_obj = obj                       

        best_child = self.flip(best_locs)                

        return best_child, best_child.fitness, n_childs

    def mutate_rls(self, l, rng=np.random.default_rng()):
        """
        generate a child by flipping exactly l bits
        Return: child, its fitness        
        """
        assert l>=0

        if l==0:
            return self, self.fitness, 0

        locs = rng.choice(self.n, size=l, replace=False) 
        child = self.flip(locs)

        return child, child.fitness       

    def crossover(self, xprime, p, n_childs, 
                    include_xprime=True, count_different_inds_only=True,
                    rng=np.random.default_rng()):
        """
        Crossover operator:
            for each bit, taking value from x with probability p and from self with probability 1-p
        Arguments:
            x: the individual to crossover with
            p (float): in [0,1]                                                
        """
        assert p <= 1
        
        if p == 0:
            if include_xprime:
                return xprime, xprime.fitness, 0
            else:
                return self, self.fitness, 0            

        if include_xprime:
            best_obj = xprime.fitness
        else:
            best_obj = -1            
        best_locs = None

        n_evals = 0
        ls = rng.binomial(self.n, p, size=n_childs)        
        for l in ls:                   
            locs_xprime = rng.choice(self.n, l, replace=False)
            locs_x = np.full(self.n, True)
            locs_x[locs_xprime] = False
            obj = self.get_fitness_after_crossover(xprime, locs_x, locs_xprime) 
                   
            if (obj != self.fitness) and (obj!=xprime.fitness):
                n_evals += 1
            elif (not np.array_equal(xprime.data[locs_xprime], self.data[locs_xprime])) and (not np.array_equal(self.data[locs_x], xprime.data[locs_x])):            
                n_evals += 1            

            if obj > best_obj:
                best_obj = obj
                best_locs = locs_xprime
            
            
        if best_locs is not None:
            child = self.combine(xprime, best_locs)
        else:
            child = xprime

        if not count_different_inds_only:
            n_evals = n_childs

        return child, child.fitness, n_evals


class OneMax(BinaryProblem):
    """
    An individual for OneMax problem
    The aim is to maximise the number of 1 bits
    """

    def eval(self):
        self.fitness = self.data.sum()
        return self.fitness

    def is_optimal(self):
        return self.data.all()

    def get_optimal(self):
        return self.n

    def get_fitness_after_flipping(self, locs):        
        # f(x_new) = f(x) + l - 2 * sum_of_flipped_block
        return self.fitness + len(locs) - 2 * self.data[locs].sum()

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):        
        return self.data[locs_x].sum() + xprime.data[locs_xprime].sum()
        

class LeadingOne(BinaryProblem):    
    """
    An individual for LeadingOne problem
    The aim is to maximise the number of leading (and consecutive) 1 bits in the string
    """

    def eval(self):
        k = self.data.argmin()
        if self.data[k]:
            self.fitness = self.n
        else:
            self.fitness = k
        return self.fitness

    def is_optimal(self):
        return self.data.all()  

    def get_optimal(self):
        return self.n    

    def get_fitness_after_flipping(self, locs):        
        min_loc = locs.min()
        if min_loc < self.fitness:
            return min_loc
        elif min_loc > self.fitness:
            return self.fitness
        else:
            old_fitness = self.fitness
            self.data[locs] = ~self.data[locs]
            new_fitness = self.eval()            
            self.data[locs] = ~self.data[locs]
            self.fitness = old_fitness
            return new_fitness


    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        child = self.combine(xprime, locs_xprime)                
        child.eval()
        return child.fitness
        

HISTORY_LENGTH = 5

class OneLLEnv(AbstractEnv):
    """
    Environment for (1+(lbd, lbd))-GA
    for both OneMax and LeadingOne problems
    """

    def __init__(self, config) -> None:
        """
        Initialize OneLLEnv

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(OneLLEnv, self).__init__(config)        
        self.logger = logging.getLogger(self.__str__())     

        self.name = config.name   

        # whether we start at a fixed inital solution or not
        # if config.init_solution_ratio is not None, we start at a solution with f = n * init_solution_ratio
        self.init_solution_ratio = None
        if ('init_solution_ratio' in config) and (config.init_solution_ratio!=None) and (config.init_solution_ratio!='None'):            
            self.init_solution_ratio = float(config.init_solution_ratio)   
            self.logger.info("Starting from initial solution with f = %.2f * n" % (self.init_solution_ratio))     

        # name of reward function
        assert config.reward_choice in ['imp_div_evals', 'imp_div_evals_new', 'imp_minus_evals', 'minus_evals', 'imp', 'minus_evals_normalised', 'imp_minus_evals_normalised']
        self.reward_choice = config.reward_choice
        #print("Reward choice: " + self.reward_choice)        
        
        # parameters of OneLL-GA
        self.problem = globals()[config.problem]
        self.include_xprime = config.include_xprime
        self.count_different_inds_only = config.count_different_inds_only
      
        # names of all variables in a state
        self.state_description = config.observation_description
        self.state_var_names = [s.strip() for s in config.observation_description.split(',')]

        # functions to get values of the current state from histories 
        # (see reset() function for those history variables)        
        self.state_functions = []
        for var_name in self.state_var_names:
            if var_name == 'n':
                self.state_functions.append(lambda: self.n)
            elif var_name in ['lbd','lbd1','lbd2', 'p', 'c', 'lbd_choice']:
                self.state_functions.append(lambda his='history_'+var_name: vars(self)[his][-1])
            elif "_{t-" in var_name:
                k = int(var_name.split("_{t-")[1][:-1]) # get the number in _{t-<number>}
                name = var_name.split("_{t-")[0] # get the variable name (lbd, lbd1, etc)
                self.state_functions.append(lambda his='history_'+name: vars(self)[his][-k])
            elif var_name == "f(x)":
                self.state_functions.append(lambda: self.history_fx[-1])
            elif var_name == "delta f(x)":
                self.state_functions.append(lambda: self.history_fx[-1] - self.history_fx[-2])
            elif var_name == "optimal_lbd_theory":
                self.state_functions.append(lambda: np.sqrt(self.n/max(1,self.n-self.history_fx[-1])))
            else:
                raise Exception("Error: invalid state variable name: " + var_name)
        
        # names of all variables in an action        
        self.action_description = config.action_description
        self.action_var_names = [s.strip() for s in config.action_description.split(',')] # names of 
        for name in self.action_var_names:
            assert name in ['lbd', 'lbd1', 'lbd2', 'p', 'c', 'lbd_choice'], "Error: invalid action variable name: " + name
        self.action_choices = None
        if isinstance(self.action_space, gym.spaces.Discrete): # TODO: this only works if the discrete action space is 1-d
            self.action_choices = config.action_choices

        # the random generator used by OneLL-GA
        if 'seed' in config:
            seed = config.seed
        else:
            seed = None
        self.rng = np.random.default_rng(seed)   

        # for logging
        self.n_eps = 0 # number of episodes done so far
        self.outdir = None
        if 'outdir' in config:
            self.outdir = config.outdir + '/' + str(uuid.uuid4())
            #self.log_fn_rew_per_state
             
    def seed(self, seed=None):
        super(OneLLEnv, self).seed(seed)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """        
        super(OneLLEnv, self).reset_()        

        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        self.max_evals = self.instance.max_evals
        self.logger.info("n:%d, max_evals:%d" % (self.n, self.max_evals))

        # create an initial solution
        self.x = self.problem(n=self.instance.size, rng=self.rng)

        if self.init_solution_ratio:
            self.x.initialise_with_fixed_number_of_bits(int(self.init_solution_ratio * self.x.n))

        # total number of evaluations so far
        self.total_evals = 1                

        # reset histories (not all of those are used at the moment)        
        self.history_lbd = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) # either this one or the next two (history_lbd1, history_lbd2) are used, depending on our configuration
        self.history_lbd1 = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_lbd2 = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_p = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_c = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_fx = deque([self.x.fitness]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) 
        self.history_lbd_choice = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) 

        # for debug only
        self.lbds = [] 
        self.rewards = []     
        self.init_obj = self.x.fitness 
        
        return self.get_state()


    def get_state(self):
        return np.asarray([f() for f in self.state_functions])


    def get_onell_params(self, action):
        """
        Get OneLL-GA parameters (lbd1, lbd2, p and c) from an action

        Returns: lbd1, lbd2, p, c
            lbd1: float (will be converted to int in step())
                number of mutated off-springs: in range [1,n]
            lbd2: float (will be converted to int in step())
                number of crossovered off-springs: in range [1,n]
            p: float
                mutation probability
            c: float
                crossover bias
        """
        i = 0
        rs = {}
        if (not isinstance(action, np.ndarray)) and (not isinstance(action, list)):
            if self.action_choices: # TODO: only support 1-d discrete action space 
                action = self.action_choices[self.action_var_names[0]][action]
            action = [action]
        for var_name in self.action_var_names:
            if var_name == 'lbd':
                rs['lbd1'] = rs['lbd2'] = np.clip(action[i], 1, self.n)
            elif 'lbd' in var_name: # lbd1 or lbd2 
                rs[var_name] = np.clip(action[i], 1, self.n)
            else: # must be p or c
                rs[var_name] = np.clip(action[i], 0, 1)
            i+=1

        # if p and c are not set, use the default formula
        if not 'p' in rs.keys():
            rs['p'] = rs['lbd1'] / self.n
        if not 'c' in rs.keys():
            rs['c'] = 1 / rs['lbd1']

        return rs['lbd1'], rs['lbd2'], rs['p'], rs['c']
    
    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : Box
            action to execute

        Returns
        -------            
            state, reward, done, info
            np.array, float, bool, dict
        """
        super(OneLLEnv, self).step_()                
                
        fitness_before_update = self.x.fitness

        lbd1, lbd2, p, c = self.get_onell_params(action)

        # mutation phase
        xprime, f_xprime, ne1 = self.x.mutate(p, int(lbd1), self.rng)

        # crossover phase
        y, f_y, ne2 = self.x.crossover(xprime, c, int(lbd2), self.include_xprime, self.count_different_inds_only, self.rng)        
        
        # update x
        if self.x.fitness <= y.fitness:
            self.x = y
        
        # update total number of evaluations
        n_evals = ne1 + ne2
        self.total_evals += n_evals

        # check stopping criteria        
        done = (self.total_evals>=self.instance.max_evals) or (self.x.is_optimal())        
        
        # calculate reward        
        imp = self.x.fitness - fitness_before_update
        if self.reward_choice=='imp_div_evals':        
            reward = imp / n_evals
        elif self.reward_choice=='imp_div_evals_new':            
            reward = (self.x.fitness - fitness_before_update - 0.5) / n_evals
        elif self.reward_choice=='imp_minus_evals':
            reward = self.x.fitness - fitness_before_update - n_evals
        elif self.reward_choice=='minus_evals':
            reward = -n_evals
        elif self.reward_choice=='minus_evals_normalised':
            reward = -n_evals / self.max_evals            
        elif self.reward_choice=='imp_minus_evals_normalised':
            reward = (self.x.fitness - fitness_before_update - n_evals) / self.max_evals
        elif self.reward_choice=='imp':
            reward = self.x.fitness - fitness_before_update
        self.rewards.append(reward)

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_lbd1.append(lbd1)
        self.history_lbd2.append(lbd2)
        self.history_lbd.append(lbd1)
        self.history_p.append(p)
        self.history_c.append(c)
        self.history_lbd_choice.append(action)

        #print("%.2f" % (action), end='\n' if done else '\r')
        #print("steps:%5d\t evals:%5d\t lbd:%5d\t f:%5d" %(self.c_step, self.total_evals, lbd1, self.x.fitness), end='\r')
        self.lbds.append(lbd1)
        
        returned_info = {"msg": "", "values":{}}
        if done:
            self.n_eps += 1
            if hasattr(self, "env_type"):
                msg = "Env " + self.env_type + ". "
            else:
                msg = ""    
            msg += "Episode done: ep=%d; n=%d; obj=%d; init_obj=%d; evals=%d; max_evals=%d; steps=%d; lbd_min=%.1f; lbd_max=%.1f; lbd_mean=%.1f; R=%.4f" % (self.n_eps, self.n, self.x.fitness, self.init_obj, self.total_evals, self.max_evals, self.c_step, min(self.lbds), max(self.lbds), sum(self.lbds)/len(self.lbds), sum(self.rewards))      
            self.logger.info(msg) 
            returned_info['msg'] = msg
            returned_info['values'] = {'n':int(self.n), 
                                        'obj': int(self.x.fitness), 
                                        'init_obj': int(self.init_obj), 
                                        'evals': int(self.total_evals), 
                                        'max_evals': int(self.max_evals), 
                                        'steps': int(self.c_step), 
                                        'lbd_min': float(min(self.lbds)), 
                                        'lbd_max': float(max(self.lbds)), 
                                        'lbd_mean': float(sum(self.lbds)/len(self.lbds)), 
                                        'R': float(sum(self.rewards)), 
                                        'lbds':[float(x) for x in self.lbds], 
                                        'rewards': [float(x) for x in self.rewards]}
        
        return self.get_state(), reward, done, returned_info
            

    def close(self) -> bool:
        """
        Close Env

        No additional cleanup necessary

        Returns
        -------
        bool
            Closing confirmation
        """        
        return True


class RLSEnv(AbstractEnv):
    """
    Environment for RLS with step size
    for both OneMax and LeadingOne problems
    """

    def __init__(self, config) -> None:
        """
        Initialize RLSEnv

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(RLSEnv, self).__init__(config)        
        self.logger = logging.getLogger(self.__str__())     

        self.name = config.name   
        
        # parameters of RLS
        self.problem = globals()[config.problem]                

        # the random generator used by RLS
        if 'seed' in config:
            seed = config.seed
        else:
            seed = None
        self.rng = np.random.default_rng(seed)   

        # for logging
        self.n_eps = 0 # number of episodes done so far        
             

    def reset(self):
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """        
        super(RLSEnv, self).reset_()        

        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        self.max_evals = self.instance.max_evals
        self.logger.info("n:%d, max_evals:%d" % (self.n, self.max_evals))

        # create an initial solution
        self.x = self.problem(n=self.instance.size, rng=self.rng)

        # total number of evaluations so far
        self.total_evals = 1                        

        # for debug only
        self.ks = [] 
        self.rewards = []     
        self.init_obj = self.x.fitness 
        
        return self.get_state()


    def get_state(self):
        return np.asarray([self.n, self.x.fitness])
    
    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : Box
            action to execute

        Returns
        -------            
            state, reward, done, info
            np.array, float, bool, dict
        """
        super(RLSEnv, self).step_()     

        k = int(action[0])           
                
        # for logging
        fitness_before_update = self.x.fitness
        
        # flip k bits
        y, f_y, n_evals = self.x.mutate_rls(k, self.rng)         
        
        # update x
        if self.x.fitness <= y.fitness:
            self.x = y
        
        # update total number of evaluations        
        self.total_evals += n_evals

        # check stopping criteria        
        done = (self.total_evals>=self.instance.max_evals) or (self.x.is_optimal())        
        
        # calculate reward        
        #reward = self.x.fitness - fitness_before_update - n_evals
        reward = (self.x.fitness - fitness_before_update)/n_evals
        self.rewards.append(reward)

        #print("%.2f" % (action), end='\n' if done else '\r')
        #print("steps:%5d\t evals:%5d\t lbd:%5d\t f:%5d" %(self.c_step, self.total_evals, lbd1, self.x.fitness), end='\r')
        self.ks.append(k)
        
        if done:
            self.n_eps += 1
            if hasattr(self, "env_type"):
                msg = "Env " + self.env_type + ". "
            else:
                msg = ""            
            self.logger.info(msg + "Episode done: ep=%d; n=%d; obj=%d; init_obj=%d; evals=%d; steps=%d; k_min=%d; k_max=%d; k_mean=%.3f; R=%.1f" % (self.n_eps, self.n, self.x.fitness, self.init_obj, self.total_evals, self.c_step, min(self.ks), max(self.lbksds), sum(self.ks)/len(self.ks), sum(self.rewards)))                       
        
        return self.get_state(), reward, done, {}    
            

    def close(self) -> bool:
        """
        Close Env

        No additional cleanup necessary

        Returns
        -------
        bool
            Closing confirmation
        """        
        return True
