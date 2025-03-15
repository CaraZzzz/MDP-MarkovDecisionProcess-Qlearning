

import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        V = np.copy(initialV)
        iterId = 0
        epsilon = float('inf')

        # Continue until convergence or max iterations reached
        while iterId < nIterations and epsilon >= tolerance:
            # Store old value function for convergence check
            oldV = np.copy(V)
            
            # Compute Q-values for all state-action pairs
            Q = np.zeros((self.nActions, self.nStates))
            for a in range(self.nActions):
                # Q(s,a) = R(s,a) + gamma * sum(T(s'|s,a) * V(s'))
                Q[a] = self.R[a] + self.discount * self.T[a].dot(oldV)
            
            # Update V(s) = max_a Q(s,a)
            V = np.max(Q, axis=0)
            
            # Compute maximum difference for convergence check
            epsilon = np.max(np.abs(V - oldV))
            iterId += 1
        
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)

        # Compute Q-values for all state-action pairs
        Q = np.zeros((self.nActions, self.nStates))
        for a in range(self.nActions):
            # Q(s,a) = R(s,a) + gamma * sum(T(s'|s,a) * V(s'))
            Q[a] = self.R[a] + self.discount * self.T[a].dot(V)
        
        # For each state, choose the action that maximizes Q-value
        policy = np.argmax(Q, axis=0)

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)

        # Initialize system of linear equations components
        # For the equation (I - γT^π)V = R^π
        
        # Create identity matrix
        I = np.eye(self.nStates)
        
        # Build T^π and R^π by selecting transitions and rewards for the policy
        T_pi = np.zeros((self.nStates, self.nStates))
        R_pi = np.zeros(self.nStates)
        
        # For each state, get the transition and reward for the action specified by policy
        for s in range(self.nStates):
            T_pi[s] = self.T[policy[s]][s]
            R_pi[s] = self.R[policy[s]][s]
        
        # Solve the system of linear equations (I - γT^π)V = R^π
        # V = (I - γT^π)^(-1) R^π
        V = np.linalg.solve(I - self.discount * T_pi, R_pi)
    

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        V = np.zeros(self.nStates)
        iterId = 0

        # Initialize variables
        policy = np.copy(initialPolicy)
        iterId = 0
        
        while iterId < nIterations:
            # Policy evaluation
            V = self.evaluatePolicy(policy)
            
            # Policy improvement
            newPolicy = self.extractPolicy(V)
            
            # Check if policy has converged
            if np.array_equal(policy, newPolicy):
                break
                
            policy = newPolicy
            iterId += 1
    
        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        # Initialize variables
        V = np.copy(initialV)
        iterId = 0
        epsilon = float('inf')
        
        # Build T^π and R^π by selecting transitions and rewards for the policy
        T_pi = np.zeros((self.nStates, self.nStates))
        R_pi = np.zeros(self.nStates)
        
        # For each state, get the transition and reward for the action specified by policy
        for s in range(self.nStates):
            T_pi[s] = self.T[policy[s]][s]
            R_pi[s] = self.R[policy[s]][s]
        
        # Iterate until convergence or max iterations reached
        while iterId < nIterations and epsilon >= tolerance:
            # Store old value function for convergence check
            oldV = np.copy(V)
            
            # Update V using Bellman equation: V^π = R^π + γT^π V^π
            V = R_pi + self.discount * T_pi.dot(oldV)
            
            # Compute maximum difference for convergence check
            epsilon = np.max(np.abs(V - oldV))
            iterId += 1
        

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        # Initialize variables
        policy = np.copy(initialPolicy)
        V = np.copy(initialV)
        iterId = 0
        epsilon = float('inf')
        
        # Continue until convergence or max iterations reached
        while iterId < nIterations and epsilon >= tolerance:
            # Store old value function for convergence check
            oldV = np.copy(V)
            
            # Partial policy evaluation
            [V, _, _] = self.evaluatePolicyPartially(policy, V, 
                                                    nIterations=nEvalIterations, 
                                                    tolerance=tolerance)
            
            # Policy improvement
            newPolicy = self.extractPolicy(V)
            
            # Update policy
            policy = newPolicy
            
            # Compute maximum difference for convergence check
            epsilon = np.max(np.abs(V - oldV))
            iterId += 1
    

        return [policy,V,iterId,epsilon]
        
