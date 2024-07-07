import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def create_video():
    img_array = []
    for i in range(len(os.listdir('results/'))):
        img = cv2.imread('results/' + str(i+1) + '.png')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('results/dqn_simulation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
window_size = 5

# Calculate the moving average and standard deviation
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def moving_std(data, window_size):
    return np.sqrt(moving_average(np.square(data), window_size) - np.square(moving_average(data, window_size)))

def plot_metrics(steps_total, rewards, loss_total, eps_total, final_height, final_velocity, disturbance):
    
    
    # Steps
    plt.figure(figsize=(12,5))
    plt.title("Steps Number to Dock Safely")
    plt.plot(steps_total, alpha=0.6, color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Total Steps')
    plt.savefig('Steps.png')
    with open('steps_data.csv', mode='w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(['Steps'])
     for steps in steps_total:
        writer.writerow([steps])
    
    
    # Rewards
    plt.figure(figsize=(12,5))
    avg_rewards = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards)-window_size+1)]
    std_dev_reward = [np.std(rewards[i:i+window_size]) for i in range(len(rewards)-window_size+1)]
    x_values_reward = np.arange(window_size-1, len(rewards))
    plt.plot(x_values_reward, avg_rewards, label=f'Average Reward (Window size = {window_size})', color='b')
    plt.fill_between(x_values_reward, np.array(avg_rewards) - np.array(std_dev_reward), np.array(avg_rewards) + np.array(std_dev_reward), color='b', alpha=0.2)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('Reward.png')
    plt.title("Reward")
    with open('reward_data.csv', mode='w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(['Episode', 'Average Reward', 'Standard Deviation'])
     for i in range(len(x_values_reward)):
        writer.writerow([x_values_reward[i], avg_rewards[i], std_dev_reward[i]])
   
    
    # Loss
    plt.figure(figsize=(12,5))
    # Compute the moving average and standard deviation
    loss_total = [x for x in loss_total if x is not None]
    avg_loss = [np.mean(loss_total[i:i+window_size]) for i in range(len(loss_total)-window_size+1)]
    std_dev_loss = [np.std(loss_total[i:i+window_size]) for i in range(len(loss_total)-window_size+1)]
    x_values = np.arange(window_size-1, len(loss_total))
    # Plotting the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_loss, label=f'Average Loss (Window size = {window_size})', color='r')
    plt.fill_between(x_values, np.array(avg_loss) - np.array(std_dev_loss), np.array(avg_loss) + np.array(std_dev_loss), color='r', alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')


    # epsilon
    plt.figure(figsize=(12,5))
    plt.title("epsilon")
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.plot(eps_total, alpha=0.6, color='red')
    plt.savefig('epsilon.png')

    plt.figure(figsize=(12,5))
    plt.title("Final Heights")
    plt.xlabel('Episodes')
    plt.ylabel('Height at end of episode')
    plt.plot(final_height, alpha=0.6, color='red')
    plt.savefig('Final_Heights.png')

    plt.figure(figsize=(12,5))
    plt.title("Final Velocities")
    plt.xlabel('Episodes')
    plt.ylabel('Velocity at end of episode')
    plt.plot(final_velocity, alpha=0.6, color='red')
    plt.savefig('Final_Velocities.png')

    plt.figure(figsize=(12,5))
    plt.title("Average Disturbance")
    plt.xlabel('Episodes')
    plt.ylabel('Average Disturbance')
    plt.plot(disturbance, alpha=0.6, color='red')
    plt.savefig('Average_Disturbance.png')
