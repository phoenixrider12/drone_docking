import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from gym import spaces
import os

class UAVDockingEnv:
    def __init__(self, dt=0.1, enable_disturbance=False):

        self.mass = 1.0  # UAV mass
        self.g = 9.81  # Gravitational acceleration
        self.dt = dt  # Time step

        # Inertial frame coordinates (origin at ground)
        self.ix = 0.0
        self.iy = 0.0
        self.iz = 0.0

        # Body frame coordinates (origin at center of mass)
        self.bx = 0.0
        self.by = 0.0
        self.bz = 10.0 # Initial height

        # Euler angles (roll, pitch, yaw)
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0

        # Linear velocities (x, y, z)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # Angular velocities (roll, pitch, yaw)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Linear accelerations (x, y, z)
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0 # Initially hovering

        # Forces (x, y, z)
        self.thrust = 0.0
        self.fx = self.mass * (self.ax + self.g * np.sin(self.phi) * np.cos(self.theta))
        self.fy = self.mass * (self.ay - self.g * np.cos(self.phi) * np.sin(self.theta))
        self.fz = self.mass * (self.az - self.g) + self.thrust
        self.hovering_acceleration = self.g  # Force required to hover
        
        # self.target_height = 0.0
        self.dockz = 0.0 # Docking station z position
        self.target_velocity = 0.1
        self.state = [self.bz, self.vz]  # UAV state [z, z_dot]
        # self.action_space = spaces.Discrete(3)
        self.actions = np.array([self.g-1.0, self.g, self.g+1.0])  # Actions: [downward acceleration, hover, upward acceleration]
        self.k1 = 0.5 # Reward weight for z position
        self.k2 = 1.5 # Reward weight for z velocity
        self.tolerance_height = 0.5  # Tolerance for z position
        self.enable_disturbance = enable_disturbance
        self.max_steps = 100
        self.kfdz = 6.354e-4

        self.alpha = 0.076
        self.fp = 1.0  # peak frequency
        self.gamma = 3.3
        self.sigma = 0.07
        self.num_points = 1000
        # self.dt = 0.01
        self.simulation_time = 10  # seconds
        self.scaling_factor = 2.5

    def euler_angles_to_rotation_matrix(phi, theta, psi):
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)
        psi_rad = np.radians(psi)

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi_rad), -np.sin(phi_rad)],
                        [0, np.sin(phi_rad), np.cos(phi_rad)]])

        R_y = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                        [0, 1, 0],
                        [-np.sin(theta_rad), 0, np.cos(theta_rad)]])

        R_z = np.array([[np.cos(psi_rad), -np.sin(psi_rad), 0],
                        [np.sin(psi_rad), np.cos(psi_rad), 0],
                        [0, 0, 1]])

        R = np.dot(np.dot(R_z, R_y), R_x)

        return R

    def reset(self):
        self.state = [10.0, 0.0]
        return self.state

    def dynamics(self, state, t, action):
        z, z_dot = state
        z_dot_dot = action  # Action directly affects z acceleration
        state_dot = [z_dot, z_dot_dot]
        return state_dot

    def gaussian(x, variance):
        pdf = (1 / (np.sqrt(2 * np.pi) * np.sqrt(variance))) * np.exp(-((x) ** 2) / (2 * variance))
        return pdf

    def jonswap_spectrum(self, f):
        non_zero_f = f.copy()
        non_zero_f[f == 0] = 1e-6  # Replace zero with a small value to avoid division by zero
        sigma = np.ones_like(non_zero_f) * self.sigma
        m = np.exp(-0.5 * ((non_zero_f - self.fp) / (sigma * self.fp)) ** 2)
        s = self.alpha * self.g ** 2 * non_zero_f ** (-5) * np.exp(-1.25 * (self.fp / non_zero_f) ** 4) * self.gamma ** m
        return np.abs(s)  # Ensure the spectrum values are non-negative

    # Generate random wave using JONSWAP spectrum
    def generate_random_wave(self):
        num_samples = int(self.simulation_time / self.dt)
        f = np.fft.fftfreq(self.num_points, self.dt)
        mag = np.sqrt(self.jonswap_spectrum(f))
        phase = np.random.rand(self.num_points) * 2 * np.pi
        fourier = mag * (np.cos(phase) + 1j * np.sin(phase))
        wave = np.fft.ifft(fourier)
        wave *= self.scaling_factor
        return np.real(wave)[:num_samples]

    def disturbance(self, disturbance_bound):
        samples = np.linspace(-100, 100, 500, endpoint=False)
        variance = disturbance_bound
        gaussian_values = []
        for i in range(len(samples)):
            gaussian_values.append(self.gaussian(samples[i], variance))
        return gaussian_values # Use this list of values to randomly select a disturbance and add to z pos

    def step(self, action_idx, time, step, wave):
        
        if self.enable_disturbance:
            self.dockz = wave[step - 1]
        # print('dockz:', self.dockz)
        action = self.actions[action_idx] # az
        self.az = action - self.g - (self.kfdz * self.state[0]) / self.mass
        new_state = odeint(self.dynamics, self.state, [time, time+self.dt], args=(self.az,))[-1]
        self.state = new_state # z pos and z vel
        reward = self.calculate_reward(self.state)
        done = self.is_done(step)
        
        return new_state, reward, done, {}

    def calculate_reward(self, state):
        if abs(state[0] - self.dockz) > (self.bz / 2):
            reward = - self.k1 * abs(state[0] - self.dockz)
        else:
            reward = - self.k2 * abs(state[1] - self.target_velocity)
        return reward

    def is_done(self, step):
        if abs(self.state[0] - self.dockz) <= self.tolerance_height or step == self.max_steps or self.state[0] < 0:
            return True
        else:
            return False

    def initialize_render(self):
        self.drone = plt.imread("../drone.png")
        self.base = plt.imread("../baseplane.png")
        self.dronebox = OffsetImage(self.drone, zoom = 0.15)   
        self.basebox = OffsetImage(self.base, zoom = 0.15)
        
        self.fig, self.ax = plt.subplots(figsize=(12,5))
        self.ax.set_title('Episode: 500')
        self.ax.set_xlabel('X coordinate (m)')
        self.ax.set_ylabel('Height (m)')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0, 12)

        self.results_dir = 'results'
        if os.path.exists(self.results_dir):
            os.system('rm -rf ' + self.results_dir)
            os.mkdir(self.results_dir)

    def render(self, total_time, total_reward, step):
        self.ax.clear()

        uav_x = self.bx  # X-coordinate for the UAV (assumed to be at the center)
        uav_y = self.by  # Y-coordinate for the UAV (assumed to be at the center)
        uav_z = self.state[0]  # Z-coordinate for the UAV

        self.ax.text(-4.5, 9, 'Time: {:.3f} sec'.format(total_time))
        self.ax.text(3.5, 9, 'Reward: {:.3f}'.format(total_reward))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-1, 12)
        drone_ab = AnnotationBbox(self.dronebox, (uav_x, uav_z), frameon = False)
        base_ab = AnnotationBbox(self.basebox, (0, self.dockz), frameon = False)
        self.ax.add_artist(drone_ab)
        self.ax.add_artist(base_ab)

        # plt.pause(0.01)  # Add a small pause to allow rendering to be visible

        plt.savefig('results/' + str(step) + '.png')
