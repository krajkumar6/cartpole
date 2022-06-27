import random

class replay_buffer:
    def __init__(self,capacity):
        self.capacity = capacity
        # creating a replay buffer as a dictionary of (state,action,reward,next_state,done) tuples of size capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        # self.replay_buffer = {}
        self.idx = 0
    
    def add_exper(self,state,action,reward,next_state,done):
        # cycle the index to be always less than the total capacity and keep overriding the oldest value in the dictionary
        if(len(self.states) < self.capacity):
            self.states.append(state)
            self.actions.append(action) 
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        else:
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward
            self.next_states[self.idx] = next_state
            self.dones[self.idx] = done
            self.idx+=1
        #recycle the index when capacity is reached
        self.idx = self.idx % (self.capacity - 1)
            
    def fill_buffer(self, env):
       
        expers = 1
        # getting the first state
        rand_action = env.action_space.sample()
        obs,reward,done,info = env.step(rand_action)
        
        while expers <= self.capacity:
            if done:
                env.reset()
                rand_action = env.action_space.sample()
                obs,reward,done,info = env.step(rand_action)
            rand_action = env.action_space.sample()
            next_obs,reward,done,info = env.step(rand_action)
            self.add_exper(obs,rand_action,reward,next_obs,done)
            expers+=1
            obs = next_obs        
   
    # sample the replay buffer of size sample_size
    def sample(self,sample_size):
        # input sample_size - size of the sample to be retrieved from replay buffer
        
        # generate a random set of indices that can be used to pull SARS tuples from replay memory
        sample = random.sample(range(self.capacity),sample_size)
        return(sample)
    
    def test_rep_buffer_sample(self,sample):
        for i in range(len(sample)):
            print('states ', self.states[i])
            print('actions ', self.actions[i])
            print('rewards ', self.rewards[i])
            print('next_states ', self.next_states[i])
            print('dones ', self.dones[i])
    
  
    
        