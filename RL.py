import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random )
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
        policy = np.zeros(self.mdp.nStates,int)

        # Initialize Q-function
        Q = np.copy(initialQ)
        
        # Learning rate (you might want to adjust this)
        alpha = 0.1
        
        # For each episode
        for episode in range(nEpisodes):
            # Initialize state
            state = s0
            
            # For each step
            for step in range(nSteps):
                # Choose action using exploration strategy
                if epsilon > 0:  # ε-greedy exploration
                    if np.random.random() < epsilon:
                        # Random action
                        action = np.random.randint(self.mdp.nActions)
                    else:
                        # Greedy action
                        action = np.argmax(Q[:,state])
                elif temperature > 0:  # Boltzmann exploration
                    # Compute Boltzmann probabilities
                    q_temp = Q[:,state] / temperature
                    # Subtract max for numerical stability
                    q_temp = q_temp - np.max(q_temp)
                    probs = np.exp(q_temp) / np.sum(np.exp(q_temp))
                    # Choose action according to Boltzmann distribution
                    action = np.random.choice(self.mdp.nActions, p=probs)
                else:  # Greedy policy
                    action = np.argmax(Q[:,state])
                
                # Take action, observe reward and next state
                [reward, nextState] = self.sampleRewardAndNextState(state, action)
                
                # Q-learning update
                # Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                Q[action,state] = Q[action,state] + alpha * (
                    reward + 
                    self.mdp.discount * np.max(Q[:,nextState]) - 
                    Q[action,state]
                )
                
                # Update state
                state = nextState
        
        # Extract policy from Q-function
        policy = np.argmax(Q, axis=0)
    

        return [Q,policy]    