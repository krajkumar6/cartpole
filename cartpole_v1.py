from random import seed
import gym
import matplotlib.pyplot as plt
import numpy as np
from replay_buffer import replay_buffer

#create environment
env = gym.make('CartPole-v1')
obs,info  = env.reset(seed = 42, return_info = True)
done = False

#create replay buffer
rep_buffer = replay_buffer(1000)

#fill up replay buffer
rep_buffer.fill_buffer(env)

#random sampling of replay buffer
sample = rep_buffer.sample(5)
print('random sample: ',sample)
rep_buffer.test_rep_buffer_sample(sample)


env.close
#create environment <end>

#create target network
#Training the agent
#Playing the target policy
