import numpy as np
class Rat(object):

    def __init__(self,**kwargs):
        # Initialize rat object
        self.params = {}
        self.state = 0 # state 1 corresponds to the "wandering" state
        self.Q = []
        for key,value in kwargs.items():
            self.params[key] = value
        assert('Nsolutions' in self.params)
        self.Q.append([0 for i in range(0,self.params['Nsolutions']+1)])
        for i in range(0,self.params['Nsolutions']):
            self.Q.append([0 for i in range(0,2)])
        assert('policyType' in self.params) # make sure a policy type was given
        self.policyType = self.params['policyType']
        assert('updateType' in self.params) # make sure a methed for updating Q is given
        self.updateType = self.params['updateType']
        if (self.updateType == 'SARSA' or self.updateType == 'q-learning'):
            assert('alpha' in self.params)
            assert('gamma' in self.params)
        if (self.policyType == 'e-greedy'):
            assert('epsilon' in self.params) # if the policy type was e-greedy, make sure epsilon is given


    def act(self,state,time):
        if (self.policyType == 'e-greedy'):
            r = np.random.rand()
            if (r < self.params['epsilon']):
                a = np.floor(np.random.rand()*len(self.Q[state]))
            else:
                a = np.argmax(self.Q[state])
            return a
        elif (self.policyType == 'UCB'):
            assert('c' in self.params)
            c = self.params['c']
            a = np.argmax(self.Q[state] + c*np.sqrt(np.log(time)/self.Nactions[state]))
            return a


    def update(self,state1,actions,reward,state2):
        alpha = self.params['alpha']
        gamma = self.params['gamma']
        if (self.updateType == 'SARSA'):
            assert(len(actions) == 2)
            self.Q[state][actions[0]] += alpha*(reward + gamma*self.Q[state2][actions[1]] - self.Q[state][actions[0]])
        elif (self.updateType == 'q-learning'):
            assert(len(actions) == 1)
            self.Q[state][actions[0]] += alpha*(reward + gamma*np.max(self.Q[state2]) - self.Q[state][actions[0]])
            self.state = state2

class StaySwitchSession(object):

    def __init__(self,rat,**kwargs):
        self.rat = rat
        self.timestep = 1
        for key,value in kwargs.items():
            self.params[key] = value
        self.states = [0]
        self.actions = []
        self.rewards = []

    def step(self):
        solution_rewards = self.params['solution_rewards']
        ratState = self.rat.state
        a = rat.act(ratState,self.timestep)
        if (ratState == 0):
            newState = ratState + a
            if (a == 0):
                r = self.params['wait_cost']
            else:
                r = self.params['move_cost']
        else:
            if (a == 0):
                r = solution_rewards[ratState-1]
                newState = ratState
            else:
                r = self.params['move_cost']
                newState = 0
        self.rat.update(ratState,a,r,newState)
        self.states.append(newState)
        self.actions.append(a)
        self.rewards.append(r)
        self.timestep+=1

class StaySwitchExperiment(object):

    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            self.params[key] = value
        assert('Nsessions' in self.params)
        assert('solutions_order' in self.params)
        assert('solution_values' in self.params)
