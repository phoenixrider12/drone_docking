# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:11:42 2023

@author: AliMohamedAli
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to generate JONSWAP spectrum
def jonswap_spectrum(f, alpha, g, fp, gamma, sigma):
    non_zero_f = f.copy()
    non_zero_f[f == 0] = 1e-6  # Replace zero with a small value to avoid division by zero
    sigma = np.ones_like(non_zero_f) * sigma
    m = np.exp(-0.5 * ((non_zero_f - fp) / (sigma * fp)) ** 2)
    s = alpha * g ** 2 * non_zero_f ** (-5) * np.exp(-1.25 * (fp / non_zero_f) ** 4) * gamma ** m
    return np.abs(s)  # Ensure the spectrum values are non-negative

# Generate random wave using JONSWAP spectrum
def generate_random_wave(alpha, g, fp, gamma, sigma, num_points, dt, simulation_time):
    num_samples = int(simulation_time / dt)
    f = np.fft.fftfreq(num_points, dt)
    mag = np.sqrt(jonswap_spectrum(f, alpha, g, fp, gamma, sigma))
    phase = np.random.rand(num_points) * 2 * np.pi
    fourier = mag * (np.cos(phase) + 1j * np.sin(phase))
    wave = np.fft.ifft(fourier)
    return np.real(wave)[:num_samples]

# Parameters for the JONSWAP spectrum
alpha = 0.076
g = 9.81
fp = 1.0  # peak frequency
gamma = 3.3
sigma = 0.07
num_points = 1000
dt = 0.01
simulation_time = 10  # seconds

# Generate random wave
time = np.arange(0, simulation_time, dt)
wave = 2.5*generate_random_wave(alpha, g, fp, gamma, sigma, num_points, dt, simulation_time)

# Plot the wave height against time
# plt.figure(figsize=(12, 6))
# plt.plot(time, wave, linewidth=1.5, label='Simulated Wave')
# plt.title('Simulated Random Wave using JONSWAP Spectrum', fontsize=16)
# plt.xlabel('Time (seconds)', fontsize=14)
# plt.ylabel('Wave Height(meters)', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.show()