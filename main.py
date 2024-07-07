import torch
import random
import time
import numpy as np
from env import UAVDockingEnv
from agent import QNet_Agent
from replay_buffer import ReplayBuffer
from utils import create_video, plot_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed_value = 1
torch.manual_seed(seed_value)
random.seed(seed_value)

gamma = 1
replay_mem_size = 50000
egreedy = 1
egreedy_final = 0.01
report_interval = 10
score_to_solve = 195

# Configurable Parameters
clip_error = False
double_dqn = False
dueling_dqn = False

update_target_frequency = 10
learning_rate = 0.0003
batch_size = 64
egreedy_decay = 1200
hidden_layer = 64
num_episodes = 500
enable_disturbance = True           # can be configured

memory = ReplayBuffer(replay_mem_size)
env = UAVDockingEnv(dt=0.1,enable_disturbance=enable_disturbance)

number_of_inputs = 2
number_of_outputs = 3

def train():

    agent = QNet_Agent(env, memory, number_of_inputs, hidden_layer, number_of_outputs, device, learning_rate, batch_size, gamma, update_target_frequency, egreedy_final, egreedy, egreedy_decay, double_dqn, dueling_dqn)

    steps_total = []
    loss_total = []
    eps_total = []
    rewards = []

    frames_total = 0 
    start_time = time.time()

    final_height = []
    final_velocity = []
    disturbance = []
    for i_episode in range(num_episodes):

        state = env.reset()

        step = 0
        score = 0
        wave = 10 * env.generate_random_wave()
        disturbance.append(np.mean(wave))
        while True:

            step += 1
            frames_total += 1

            epsilon = agent.calculate_epsilon(frames_total)
            action = agent.select_action(state, epsilon)
            curr_time = time.time()
            new_state, reward, done, info = env.step(action, curr_time, step, wave)
            print('state:', state, 'action:', action, 'new_state:', new_state)
            env.state = new_state
            score += reward
            # new_episode = False

            memory.push(state, action, new_state, reward, done)
            agent.optimize()
            loss = agent.optimize()
            
            state = new_state

            if done:
                steps_total.append(step)
                loss_total.append(loss)
                eps_total.append(epsilon)
                rewards.append(score)

                print("Finshing Time" , step)
                print("Number of the Eposides",i_episode)
                print("epsilon",epsilon)
                print("reward",reward)
                print("Loss Value", loss)
                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                print('\n')

                break

        final_height.append(state[0])
        final_velocity.append(state[1])

    torch.save(agent.nn, 'dueling_dqn_model_env.pth')

    print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))   # Print the average number of steps over the number of episodes
    loss_sum = 0
    count = 0
    for i in range(len(loss_total)):
      if loss_total[i] != None:
        loss_sum += loss_total[i]
        count += 1
    print('Average loss:', float(loss_sum)/float(count))

    plot_metrics(steps_total, rewards, loss_total, eps_total, final_height, final_velocity, disturbance)
    np.save('steps.npy', steps_total)
    np.save('reward.npy', rewards)
    np.save('loss.npy', loss_total)
    np.save('epsilon.npy', eps_total)
    np.save('heights.npy', final_height)
    np.save('velocities.npy', final_velocity)
    np.save('disturbance.npy', disturbance)
def simulate():

    agent = QNet_Agent(env, memory, number_of_inputs, hidden_layer, number_of_outputs, device, learning_rate, batch_size, gamma, update_target_frequency, egreedy_final, egreedy, egreedy_decay, double_dqn, dueling_dqn)
    agent.nn = torch.load('dueling_dqn_model_env.pth')

    done = False
    env.state = env.reset()
    total_reward = 0
    total_time = 0
    initial_time = time.time()
    step = 0

    wave = 10 * env.generate_random_wave()
    step_times = 0
    while not done:
        step += 1
        epsilon = 0
        step_time = time.time()
        action = agent.select_action(env.state, epsilon)
        curr_time = time.time()
        print('Time:', curr_time - step_time)
        step_times += curr_time - step_time
        new_state, reward, done, info = env.step(action, curr_time, step, wave)
        print('state:', env.state, 'action:', action, 'new_state:', new_state)
        # new_episode = False
        env.state = new_state

        zpos = env.state[0]
        xpos = env.bx
        total_reward += reward
        curr_time = time.time()
        total_time += (curr_time - initial_time)

        env.render(total_time, total_reward, step)

        if done:
            print('Time Taken:', total_time)
            print('Total Reward:', total_reward)
            print('Total Steps:', step)                
            print('Average Inference Time:', step_times/step)
            break

to_train = True      # True for training and False for evaluation

if to_train:
    train()
else:
    env.initialize_render()
    simulate()
    create_video()
