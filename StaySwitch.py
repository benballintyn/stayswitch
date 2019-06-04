import numpy as np
class Rat(object):

    def __init__(self,**kwargs):
        # Initialize rat object
        self.params = {}
        self.Q = [[0, 0],[0, 0, 0],[0, 0]]
        self.Nactions = [[0, 0],[0, 0, 0],[0, 0]]
        for key,value in kwargs.items():
            self.params[key] = value
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
        else if (self.policyType == 'UCB'):
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
        else if (self.updateType == 'q-learning'):
            assert(len(actions) == 1)
            self.Q[state][actions[0]] += alpha*(reward + gamma*np.max(self.Q[state2]) - self.Q[state][actions[0]])

