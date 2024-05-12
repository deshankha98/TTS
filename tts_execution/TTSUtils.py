import math
import random
import re

import numpy as np
import torch
from pydub import AudioSegment

OS_PATH_DELIMITER = "/"


def space2pipe(text: str):
    return re.sub('[\s]+', "|", text)


def add_gaussian_noise(sig, sr, snr=20):
    pass
def add_random_uniform_noise(sig, sr, snr=5):
    sig_power = torch.sum(torch.pow(sig, 2))/len(sig[0])
    noise_factor = math.exp((math.log((sig_power * 3)) - snr) * 0.5)
    noise = np.random.uniform(0, 1, size=(1, len(sig[0]))) * noise_factor  #### -----> power contained in noise = 1/3 * (noise_factor)^2
    noise = torch.tensor(noise)
    sig += noise
    return sig
def add_random_normal_noise(sig, sr, snr=5):
    sig_power = torch.sum(torch.pow(sig, 2)) / len(sig[0])
    noise_factor = math.exp((math.log(sig_power) - snr) * 0.5)
    noise = np.random.normal(0, noise_factor, size=(
    1, len(sig[0])))  #### -----> power contained in noise = noise_factor^2
    noise = torch.tensor(noise)
    sig += noise
    return sig

def add_silence_with_noise(sig, sr, snr=5):
    left_silence_duration = random.choice([0,0.5, 1, 2])
    right_silence_duration = random.choice([0,0.5, 1,2])
    noise_form = random.choice(["uniform", "normal"])
    left_zeros_ = torch.zeros((1, int(left_silence_duration * sr)))
    right_zeros_ = torch.zeros((1, int(right_silence_duration * sr)))
    sig = torch.cat((left_zeros_, sig, right_zeros_), dim=1)
    sig_power = torch.sum(torch.pow(sig, 2))/len(sig[0])
    if noise_form == "uniform":
        noise_factor = math.exp((math.log((sig_power * 3)) - snr) * 0.5)
        noise = np.random.uniform(0, 1, size=(1, len(sig[0]))) * noise_factor
        noise = torch.tensor(noise)
    elif noise_form == "normal":
        noise_factor = math.exp((math.log(sig_power) - snr) * 0.5)
        noise = np.random.normal(0, noise_factor, size=(
            1, len(sig[0])))  #### -----> power contained in noise = noise_factor^2
        noise = torch.tensor(noise)
    sig += noise
    return sig

def speedup_with_noise(input_file_name, snr=5):
    speedup_factors = [1.25, 1.5]
    speedup_factor = random.choice(speedup_factors)
    noise_form = random.choice(["uniform", "normal"])

    audio = AudioSegment.from_file(
        input_file_name,
        format="wav")
    sped_audio = audio.speedup(playback_speed=speedup_factor)
    raw_data = sped_audio.get_array_of_samples()
    max_val = max(raw_data)
    raw_data = np.array(raw_data, dtype=np.float32)
    raw_data = raw_data.reshape((1, len(raw_data)))
    raw_data = raw_data / max_val
    sig = torch.tensor(raw_data)
    sig_power = torch.sum(torch.pow(sig, 2)) / len(sig[0])
    if noise_form == "uniform":
        noise_factor = math.exp((math.log((sig_power * 3)) - snr) * 0.5)
        noise = np.random.uniform(0, 1, size=(1, len(sig[0]))) * noise_factor
        noise = torch.tensor(noise)
    elif noise_form == "normal":
        noise_factor = math.exp((math.log(sig_power) - snr) * 0.5)
        noise = np.random.normal(0, noise_factor, size=(
            1, len(sig[0])))  #### -----> power contained in noise = noise_factor^2
        noise = torch.tensor(noise)
    sig += noise
    return sig



