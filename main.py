import torch
import random
import time
import numpy as np
import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append( os.path.join(os.path.dirname( os.path.abspath(__file__) ) ,"./PPO/"))
from env import UAVDockingEnv
from agent import ActorNetwork, CriticNetwork, PPO_Agent
from replay_buffer import PPOMemory
from utils import create_video, plot_metrics
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

gamma = 0.99
learning_rate = 0.0003
gae_lambda = 0.95
policy_clip = 0.2
batch_size = 5
n_epochs = 4
hidden_layer = 256
steps_update_frequency = 5

def initialization(enable_disturbance):
    env = UAVDockingEnv(dt=0.1, enable_disturbance=enable_disturbance)

    agent = PPO_Agent(n_actions=len(env.actions), input_dims=2,
                  batch_size=batch_size, gamma=gamma, learning_rate=learning_rate,
                  gae_lambda=gae_lambda, policy_clip=policy_clip, n_epochs=n_epochs,
                  hidden_layer=hidden_layer, steps_update_frequency=steps_update_frequency)

    return env, agent

def training(num_training_episodes, env, agent):
    best_score = 0.0
    score_history = []
    avg_score_history = []
    endingtime_history = []
    total_steps = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    actor_loss = np.zeros((num_training_episodes, 1))
    critic_loss = np.zeros((num_training_episodes, 1))
    total_loss = np.zeros((num_training_episodes, 1))

    final_height = []
    final_velocity = []
    disturbance = []

    for i in range(num_training_episodes):
        observation = env.reset()
        done = False
        score = 0
        n_steps = 0

        wave = 10 * env.generate_random_wave()    # Initializing new wave for each episode
        disturbance.append(np.mean(wave))
        
        while True:
            # Check which is the environment
            if type(env).__name__ == 'FactoryEnv':
               mask, feasible_idxs = env.mask_fn()
            else:
                mask=None
                feasible_idxs=None

            action, prob, val = agent.choose_action(observation, mask, feasible_idxs)
            curr_time = time.time()
            n_steps += 1

            observation_, reward, done, info = env.step(action, curr_time, n_steps, wave)
            # print('state:', observation, 'action:', action, 'new state:', observation_)
            score += reward

            agent.remember(observation, action, prob, val, reward, done, mask)

            if n_steps % agent.steps_update_frequency == 0:
                al, cl, tl = agent.learn()
                learn_iters += 1
                actor_loss[i] += al
                critic_loss[i] += cl
                total_loss[i] += tl

            observation = observation_
            env.state = observation_

            if done:
                break
                print(' ')

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        total_steps.append(n_steps)

        final_height.append(env.state[0])
        final_velocity.append(env.state[1])

        actor_loss[i] /= (n_steps/agent.steps_update_frequency)
        critic_loss[i] /= (n_steps/agent.steps_update_frequency)
        total_loss[i] /= (n_steps/agent.steps_update_frequency)

        if type(env).__name__ == 'FactoryEnv':
            finishing_time = env.k
            endingtime_history.append(finishing_time)
        else:
            finishing_time = "NULL"

        if avg_score > best_score:
            best_score = avg_score

        if i%10 == 0:
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', n_steps, 'learning_steps', learn_iters, 'ending_time: ', finishing_time)

    return env, agent, score_history, avg_score_history, total_steps, actor_loss, critic_loss, total_loss, final_height, final_velocity, disturbance

def evaluation(num_testing_episodes, env, agent):
    Scores = []
    Endingtimes = []
    agent.actor.eval()
    steps = 0

    for i in range(num_testing_episodes):
        observation = env.reset()
        done = False
        score = 0

        wave = 10 * env.generate_random_wave()       # Wave initialization

        while not done:
            # Check which is the environment
            if type(env).__name__ == 'FactoryEnv':
               mask, feasible_idxs = env.mask_fn()
            else:
                mask=None
                feasible_idxs=None
            
            steps += 1

            action, _, _ = agent.choose_action(observation, mask, feasible_idxs)
            curr_time = time.time()
            observation_, reward, done, _ = env.step(action, curr_time, steps, wave)

            score += reward

            observation = observation_


        Scores.append(score)
        if type(env).__name__ == 'FactoryEnv':
            finishing_time = env.k
            Endingtimes.append(finishing_time)
        else:
            finishing_time = "NULL"

        print("Episode " + str(i) + " | score: ", score, "| endingtime: ", finishing_time)

    print("\nEvaluation Ended.\n")

    print("Average Score on "+ str(num_testing_episodes) +" episodes: " +
          str(round(np.array(Scores).mean(), 5)) + "  " + u"\u00B1" + "  " + str(round(np.array(Scores).std(), 5)))

    print("Average Endingtime on "+ str(num_testing_episodes) +" episodes: " +
          str(np.array(Endingtimes).mean()) + "  " + u"\u00B1" + "  " + str(round(np.array(Endingtimes).std(), 5)))

    return env, agent

def save_agent(agent, current_working_directory):
    PATH = get_saving_path(current_working_directory)
    agent.save_model(PATH)

def get_saving_path(current_working_directory):
    current_time = str(round(time.time()))
    PATH = os.path.join(current_working_directory, "data", "PPO", current_time)
    os.makedirs(PATH, exist_ok=True)

    return PATH

def load_agent(agent, session, current_working_directory):
    session = str(session)
    PATH = os.path.join(current_working_directory, "data", "PPO", session, "Agent")
    agent.load_model(PATH)
    return agent

def simulate_agent(env, agent):

    done = False
    state = env.reset()
    total_reward = 0
    total_time = 0
    initial_time = time.time()
    step = 0
    disturbance_bound = 1
    new_episode = True
    wave = 10 * env.generate_random_wave()    # Wave initialization

    while not done:
        # Check which is the environment
        if type(env).__name__ == 'FactoryEnv':
           mask, feasible_idxs = env.mask_fn()
        else:
            mask=None
            feasible_idxs=None

        action, _, _ = agent.choose_action(state, mask, feasible_idxs)
        curr_time = time.time()
        step += 1
        state_, reward, done, _ = env.step(action, curr_time, step, wave)
        # print('state:', state, 'action:', action, 'new state:', state_)
        new_episode = False

        state = state_

        ypos = state[0]
        xpos = 0
        total_reward += reward
        curr_time = time.time()
        total_time += (curr_time - initial_time)

        env.render(total_time, total_reward, step)
        if done:
            env.render(total_time, total_reward, step)
            print('Time Taken:', total_time)
            print('Total Reward:', total_reward)
                
            break


num_training_episodes = 500
num_testing_episodes = 10

# SAVING:
saving_flag = True           # True if you want to save the trained model
cwd = os.getcwd()

# LOADING:
loading_flag = False         # True if you want to load a pre-trained model, False if you want to train a new model
session = 1698522963

#SIMULATION
simulation_flag = False      # True if you want to simulate the trained agent and save results(video)

#DISTURBANCE
enable_disturbance = True    # True if you want to enable the disturbance


if __name__ == '__main__':

    env, agent = initialization(enable_disturbance)

    # Loading
    if loading_flag:
        try:
            agent = load_agent(agent=agent, session=session, current_working_directory=cwd)
            print("Agent weights correctly loaded")

        except:
            print("You tried to load a model of the weights that does not match the initialized network shape. (Probably loading a not dueling_dqn model having defined a dueling one")
            raise

    else:
        # Training
        env, agent, score_history, avg_score_history, total_steps, actor_loss, critic_loss, total_loss, final_height, final_velocity, disturbance = training(num_training_episodes, env, agent)
        plot_metrics(total_steps, score_history,avg_score_history, actor_loss, critic_loss, total_loss, final_height, final_velocity, disturbance)

    # Evaluation
    if evaluation_flag:
        env, agent = evaluation(num_testing_episodes, env, agent)

    # Saving
    if saving_flag:
        try:
            save_agent(agent=agent, current_working_directory=cwd)
        except:
            raise

    if simulation_flag:
        env.initialize_render()
        simulate_agent(env, agent)
        create_video()
