"""
Created on Mon May 7 20:12:40 2018

@author: nader
"""

import numpy as np
import random
#for loading the brain
import os
import torch
#module for implementing neural network
import torch.nn as nn
#contains functions for implementing neural network
import torch.nn.functional as F
#for optimizing stochastic gradient decent
import torch.optim as optim
#distencer to tensor and gradient
#import torch.autograd as autograd
from torch.autograd import Variable

# Neural Network
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # nn.Linear => all neuran in input layer are connected to hidden layer
        # fc1 , fc2 => full connections
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):# function for propagation
        # we apply rectifier fuction to hidden neurons and we should give our input connection input states
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

#Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        #max number of transitions in memory event
        self.capacity = capacity
         #list of capacity events
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        #(state1,action1,reward1),(state2,acition2,reward2) => (state1,state2),(action1,action2),(reward1,reward2)
        #we are getting random objects from memory that have size equal to batch_size
        samples = zip(*random.sample(self.memory, batch_size))
         #we finally put each of this batch to a pytorch variable whitch each one will recieve a gradient
        #we should concatenate each batch in sample with respect with first dimenstion so that in each row state action and reward corespond to same time t
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        #parameters of our model and the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #for pytorch it not only has to be torch tensor and one more dimension  that corresponds to the batch
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #go straight
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        #we feed input state to network and we get outputs which are q value for each action
        #and use softmax to get final action
        # softmax => we want to go to best action but we want to go to others too(probability for each q value)
        # state should be torch var so we wrap this which is a tensor to torch var
        #with volatile=true we wont include gradient assosiated with this state to the graph
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial()
        #cause multinomial returns a pytorch var with fake batch we need to select index [0,0]

        # m = torch.distributions.Categorical(probs)
        # action=m.sample()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #we need gather result of action which were played
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #we get max q value in next_state(0) according to all actions(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        #we use hubble loss
        td_loss = F.smooth_l1_loss(outputs, target)
         # zero_grad reinitializes the optimizer in every iteration of loop
        self.optimizer.zero_grad()
        #free memory because we go several times on the loss
        td_loss.backward(retain_variables = True)
        #.step updates th weights by backpropagating
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        #convert signal to Tensor(float) since it is input of neural network and add dimention according to batch
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #torch.longtensor converts int to long in tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            #we get 100 from each of them
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")