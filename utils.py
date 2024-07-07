import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def create_video():
    img_array = []
    for i in range(len(os.listdir('results/'))):
        img = cv2.imread('results/' + str(i+1) + '.png')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('results/ppo_simulation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

window_size = 5

# Calculate the moving average and standard deviation
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def moving_std(data, window_size):
    return np.sqrt(moving_average(np.square(data), window_size) - np.square(moving_average(data, window_size)))

def plot_metrics(total_steps, score_history, mean_score_history, actor_loss, critic_loss, total_loss, final_height, final_velocity, disturbance):

    np.save('steps.npy', total_steps)
    np.save('score.npy', score_history)
    np.save('mean_score.npy', mean_score_history)
    np.save('actor_loss.npy', actor_loss)
    np.save('critic_loss.npy', critic_loss)
    np.save('total_loss.npy', total_loss)
    np.save('final_height.npy', final_height)
    np.save('final_velocity.npy', final_velocity)
    np.save('disturbances.npy', disturbance)

    plt.figure(figsize=(12,5))
    plt.title("Steps Number to Dock Safely")
    plt.plot(total_steps, alpha=0.6, color='green')
    plt.savefig('Total_Steps.png')

    plt.figure(figsize=(12,5))
    plt.title("Average Disturbance")
    plt.plot(disturbance, alpha=0.6, color='green')
    plt.savefig('Average_Disturbance.png')

    plt.figure(figsize=(12,5))
    plt.title("Agent Reward")
    avg_score_history = [np.mean(score_history[i:i+window_size]) for i in range(len(score_history)-window_size+1)]
    std_dev_score_history = [np.std(score_history[i:i+window_size]) for i in range(len(score_history)-window_size+1)]
    x_values_score_history = np.arange(window_size-1, len(score_history))
    plt.plot(x_values_score_history, avg_score_history, label=f'Average Reward (Window size = {window_size})', color='b')
    plt.fill_between(x_values_score_history, np.array(avg_score_history) - np.array(std_dev_score_history), np.array(avg_score_history) + np.array(std_dev_score_history), color='b', alpha=0.2)
    plt.savefig('Agent_Reward.png')

    plt.figure(figsize=(12,5))
    plt.title("Average Reward")
    avg_mean_score_history = [np.mean(mean_score_history[i:i+window_size]) for i in range(len(mean_score_history)-window_size+1)]
    std_dev_mean_score_history = [np.std(mean_score_history[i:i+window_size]) for i in range(len(mean_score_history)-window_size+1)]
    x_values_mean_score_history = np.arange(window_size-1, len(mean_score_history))
    plt.plot(x_values_mean_score_history, avg_mean_score_history, label=f'Average mean_score_history (Window size = {window_size})', color='b')
    plt.fill_between(x_values_mean_score_history, np.array(avg_mean_score_history) - np.array(std_dev_mean_score_history), np.array(avg_mean_score_history) + np.array(std_dev_mean_score_history), color='b', alpha=0.2)
    plt.savefig('Average_Agent_Reward.png')

    plt.figure(figsize=(12,5))
    plt.title("Actor Loss")
    avg_actor_loss = [np.mean(actor_loss[i:i+window_size]) for i in range(len(actor_loss)-window_size+1)]
    std_dev_actor_loss = [np.std(actor_loss[i:i+window_size]) for i in range(len(actor_loss)-window_size+1)]
    x_values_actor_loss = np.arange(window_size-1, len(actor_loss))
    plt.plot(x_values_actor_loss, avg_actor_loss, label=f'Average actor_loss (Window size = {window_size})', color='b')
    plt.fill_between(x_values_actor_loss, np.array(avg_actor_loss) - np.array(std_dev_actor_loss), np.array(avg_actor_loss) + np.array(std_dev_actor_loss), color='b', alpha=0.2)
    plt.savefig('Actor_Loss.png')

    plt.figure(figsize=(12,5))
    plt.title("Critic Loss")
    avg_critic_loss = [np.mean(critic_loss[i:i+window_size]) for i in range(len(critic_loss)-window_size+1)]
    std_dev_critic_loss = [np.std(critic_loss[i:i+window_size]) for i in range(len(critic_loss)-window_size+1)]
    x_values_critic_loss = np.arange(window_size-1, len(critic_loss))
    plt.plot(x_values_critic_loss, avg_critic_loss, label=f'Average critic_loss (Window size = {window_size})', color='b')
    plt.fill_between(x_values_critic_loss, np.array(avg_critic_loss) - np.array(std_dev_critic_loss), np.array(avg_critic_loss) + np.array(std_dev_critic_loss), color='b', alpha=0.2)
    plt.savefig('Critic_Loss.png')

    plt.figure(figsize=(12,5))
    plt.title("Total Loss")
    avg_total_loss = [np.mean(total_loss[i:i+window_size]) for i in range(len(total_loss)-window_size+1)]
    std_dev_total_loss = [np.std(total_loss[i:i+window_size]) for i in range(len(total_loss)-window_size+1)]
    x_values_total_loss = np.arange(window_size-1, len(total_loss))
    plt.plot(x_values_total_loss, avg_total_loss, label=f'Average total_loss (Window size = {window_size})', color='b')
    plt.fill_between(x_values_total_loss, np.array(avg_total_loss) - np.array(std_dev_total_loss), np.array(avg_total_loss) + np.array(std_dev_total_loss), color='b', alpha=0.2)
    plt.savefig('Total_Loss.png')
    
    plt.figure(figsize=(12,5))
    plt.title("Final Heights")
    plt.plot(final_height, alpha=0.6, color='red')
    plt.savefig('Final_Heights.png')

    plt.figure(figsize=(12,5))
    plt.title("Final Velocities")
    plt.plot(final_velocity, alpha=0.6, color='red')
    plt.savefig('Final_Velocities.png')
