import random

class ReplayBuffer(object):    # doesnot inherit  from anything
    def __init__(self, capacity):  # define the paramters of the class
        self.capacity = capacity   # capacity of the memory
        self.memory = []           # initalzie memeory array
        self.position = 0          # initalzie position to zero , it is used to track input infromation pushed in the memory
 
    def push(self, state, action, new_state, reward, done):    # push all the information of the each epo into the memory
        transition = (state, action, new_state, reward, done)   # define the transtion varabile as a LIST
        
        if self.position >= len(self.memory):    # when the capacity is full we take other transtion , we over read the exisiting one
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = ( self.position + 1 ) % self.capacity   # to increment the postion +1 and return to zero when the capcity is full
        
        
    def sample(self, batch_size):     # sample random information from the memory
        return zip(*random.sample(self.memory, batch_size))  # random sampling from memory , and how many entries is batch size
        # our data is stored in order in a list , we need to use zip* to categorize it, arrange the frist element of each list together
        
        
    def __len__(self):   # to show the current memory usage
        return len(self.memory)  # return the length of the memeory