import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time
import matplotlib.pyplot as plt
import os
import glob
import cv2

class NeuralNetwork(nn.Module):
    def __init__(self, inputs, hl, outputs):
        super(NeuralNetwork, self).__init__()
        self.number_of_inputs = inputs
        self.hidden_layer = hl
        self.number_of_outputs = outputs
        self.dueling_dqn = False
        self.double_dqn = False
        self.linear1 = nn.Linear(self.number_of_inputs,self.hidden_layer)
        self.linear2 = nn.Linear(self.hidden_layer,self.hidden_layer)
        self.linear3 = nn.Linear(self.hidden_layer,self.hidden_layer)
        self.linear4 = nn.Linear(self.hidden_layer,self.hidden_layer)
        self.linear5 = nn.Linear(self.hidden_layer,self.number_of_outputs)
        
        self.advantage = nn.Linear(self.hidden_layer,self.number_of_outputs)
        self.value = nn.Linear(self.hidden_layer,1)
        
        self.activation = nn.Tanh()
        #self.activation = nn.ReLU()
        
    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)
        output2 = self.activation(output2)
        output3 = self.linear3(output2)
        output3 = self.activation(output3)
        output4 = self.linear4(output3)
        output4 = self.activation(output4)
        output5 = self.linear5(output4)
        
        if self.dueling_dqn: 
         output_advantage = self.advantage(output4)
         output_value = self.value(output4)
        
         output_final = output_value + output_advantage - output_advantage.mean()
        else:
         output_final = output5
        return output_final

class QNet_Agent(object):
    def __init__(self, env, memory, inputs, hl, outputs, device, lr, bs, gamma, utf, egreedy_final, egreedy, egreedy_decay):
        self.env = env
        self.inputs = inputs
        self.hidden_layer = hl
        self.outputs = outputs
        self.device = device
        self.lr = lr
        self.batch_size = bs
        self.memory = memory
        self.gamma = gamma
        self.update_target_frequency = utf
        self.egreedy_final = egreedy_final
        self.egreedy = egreedy
        self.egreedy_decay = egreedy_decay
        self.clip_error = False
        self.nn = NeuralNetwork(self.inputs, self.hidden_layer, self.outputs).to(self.device)
        self.target_nn = NeuralNetwork(self.inputs, self.hidden_layer, self.outputs).to(self.device)

        self.loss_func = nn.MSELoss().to(self.device)
        #self.loss_func = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=self.lr)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.update_target_counter = 0

    def calculate_epsilon(self, steps_done):
        epsilon = self.egreedy_final + (self.egreedy - self.egreedy_final) * math.exp(-1. * steps_done / self.egreedy_decay )
        return epsilon
        
    def select_action(self, state, epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                state = torch.Tensor(state).to(self.device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()        
        else:
            action = random.choice([0,1,2])
        # print('action:', action)
        return action
    
    def optimize(self):   # see if the current size of the memory is less than the batch size 
        if (len(self.memory) < self.batch_size):
            return
        state, action, new_state, reward, done = self.memory.sample(self.batch_size)
        state = torch.Tensor(state).to(self.device)
        new_state = torch.Tensor(new_state).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)
        # we have 32 done values since we have a batch of 32

        if self.nn.double_dqn:
            new_state_indexes = self.nn(new_state).detach() # index of actions from the leanring network not the target 
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]  # get the index of the best action from the leanring netwrok
            
            new_state_values = self.target_nn(new_state).detach()
            # we  merge the values from the target network and index from the leanring network 
            # sequeez and unsequeez for just tensor dim matching 
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()   #  detach only forwad path of NN
            max_new_state_values = torch.max(new_state_values, 1)[0]
            
        target_value = reward + ( 1 - done ) * self.gamma * max_new_state_values  # the value of the best state
        # gather is for collecting a certain dim of tensors

        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        self.loss = self.loss_func(predicted_value, target_value)
        # print('loss:', self.loss)
        self.loss_value = self.loss.item()
    
        self.optimizer.zero_grad()
        self.loss.backward()
        
        if self.clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.update_target_counter % self.update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        self.update_target_counter += 1
        
        return self.loss_value 
