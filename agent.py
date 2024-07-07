import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch as T
import random
import os
from pandas import Categorical
from replay_buffer import PPOMemory
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, hidden_layer):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(input_dims, hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer, hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, mask=None):
        logits = self.actor(state)
        # print(logits)

        masked_logits = self.mask_logits(logits, mask)

        dist = Categorical(logits)

        return dist


    def mask_logits(self, logits, mask=None):

        #--- DATA TYPE CONVERSION
        if isinstance(mask, np.ndarray):
            mask = T.from_numpy(mask).to(bool)
        else:
            mask = mask

        # print("ACTION MASK: ", mask)
        # print("LOGITS BEFORE MASKING : ", logits)

        #--- LOGITS MASKING
        # Define torch minimum floating point
        mask_value = T.tensor(T.finfo(logits.dtype).min, dtype=logits.dtype)

        # If no mask is passed
        if mask is None:
            masked_logits = logits

        # If mask is passed
        else:
            # If no batch size logits
            if len(logits.size()) == 1:
                masked_logits = T.where(mask, logits, mask_value)
            # If batch size logits
            else:
                masked_logits = T.where(mask.to(bool), logits, mask_value)


        # print("LOGITS AFTER MASKING: ", masked_logits)
        return masked_logits


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, hidden_layer):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(input_dims, hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer, hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value



class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma, learning_rate, gae_lambda,
            policy_clip, batch_size, n_epochs, hidden_layer, steps_update_frequency):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.steps_update_frequency = steps_update_frequency

        self.actor = ActorNetwork(n_actions, input_dims, learning_rate, hidden_layer)
        self.critic = CriticNetwork(input_dims, learning_rate, hidden_layer)
        self.memory = PPOMemory(batch_size)

        # self.actor_loss = []
        # self.critic_loss = []
        # self.total_loss = []

    def remember(self, state, action, probs, vals, reward, done, mask=None):
        self.memory.store_memory(state, action, probs, vals, reward, done, mask)


    def choose_action(self, observation, mask=None, feasible_idxs=None):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state, mask)

        value = self.critic(state)

        action = dist.sample()

        # Double check on action feasiblity
        if feasible_idxs is not None:
            counter = 0
            while not (action.item() in feasible_idxs):
                # Try search for an action
                if counter < 5000:
                    action = dist.sample()
                    counter += 1
                # If can't find it, take a random one within the feasibles
                else:
                    # print("RANDOMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
                    action = T.tensor(random.choice(feasible_idxs))




        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        #--- EPOCHS LOOP
        for _ in range(self.n_epochs):

            #--- Sample from memory
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, masks_arr, batches = self.memory.generate_batches()

            #--- Compute advantages
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            #--- Optimization step
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                if len(self.memory.masks)>0:
                    masks = T.tensor(masks_arr[batch]).to(self.actor.device)
                else:
                    masks = None

                dist = self.actor(states, masks)

                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                actor_loss_val = actor_loss.cpu().detach().numpy()
                # self.actor_loss.append(actor_loss_val)

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                critic_loss_val = critic_loss.cpu().detach().numpy()
                # self.critic_loss.append(critic_loss_val)

                total_loss = actor_loss + 0.5*critic_loss
                total_loss_val = total_loss.cpu().detach().numpy()
                # self.total_loss.append(total_loss_val)

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

        return actor_loss_val, critic_loss_val, total_loss_val

    def save_model(self, PATH):
        print("... saving models ...")
        path = os.path.join(PATH, "Agent")
        os.makedirs(path, exist_ok=True)

        actor_path = os.path.join(path, "Actor")
        critic_path = os.path.join(path, "Critic")

        T.save(self.actor.state_dict(), actor_path)
        T.save(self.critic.state_dict(), critic_path)

    def load_model(self, PATH):
        actor_path = os.path.join(PATH, "Actor")
        critic_path = os.path.join(PATH, "Critic")

        self.actor.load_state_dict(T.load(actor_path))
        self.critic.load_state_dict(T.load(critic_path))
