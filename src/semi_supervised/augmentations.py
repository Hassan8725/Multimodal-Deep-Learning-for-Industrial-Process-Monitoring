import torch
import numpy as np
from scipy.interpolate import CubicSpline
import random


class Noise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

def apply_gaussian_noise(data, mean=0, std=0.01):
    """
    Applies Gaussian noise to each time series in the data batch.
    
    Parameters:
    data (torch.Tensor or np.ndarray): A batch of time series data with shape (batch_size, 1350, 1, 150).
    mean (float): Mean of the Gaussian noise.
    std (float): Standard deviation of the Gaussian noise.
    
    Returns:
    torch.Tensor: The batch of time series data with Gaussian noise added.
    """
    # Check if the input data is a NumPy array and convert it to a PyTorch tensor if necessary
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    # Initialize the Gaussian Noise
    GaussianNoise = Noise(mean, std)
    
    # Apply Gaussian noise to each time series in the batch
    noisy_data = torch.zeros_like(data)
    for i in range(data.size(0)):  # Iterating over batch size
        for j in range(data.size(1)):  # Iterating over 1350 series
            noisy_data[i, j] = GaussianNoise(data[i, j])
    
    return noisy_data


def DA_Scaling(X, sigma=0.001):
    # X has shape (batch_size, 1, num_features), for example (32, 1, 150)
    batch_size, _, num_features = X.shape

    # Generate a scaling factor for each feature
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, num_features))

    # Expand the scaling factor to match the batch size and the additional dimension
    scalingFactor = scalingFactor.reshape(1, 1, num_features)
    myNoise = np.tile(scalingFactor, (batch_size, 1, 1))

    # Apply the scaling
    return X * myNoise

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    num_samples, _, num_features = X.shape
    
    xx = np.linspace(0, num_samples - 1, num=knot + 2)  # points on x axis
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, num_features))  # random y values for each feature
    
    curves = np.zeros((num_samples, num_features))
    for feature in range(num_features):
        cs = CubicSpline(xx, yy[:, feature])
        curves[:, feature] = [cs(x) for x in range(num_samples)]
    return curves

def DA_MagWarp(X, sigma=0.2, knot=4):
    curves = GenerateRandomCurves(X, sigma, knot)  # Generate curves for each feature
    return X * curves[:, np.newaxis, :]

def DistortTimesteps(X, sigma=0.2, knot=4):
    time_curves = GenerateRandomCurves(X, sigma, knot)
    tt_cum = np.cumsum(time_curves, axis=0)  # cumulative sum to simulate "time"

    # Normalize to stretch/compress to the original length
    for i in range(tt_cum.shape[1]):
        tt_cum[:, i] = tt_cum[:, i] * (len(X) - 1) / tt_cum[-1, i]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2, knot=4):
    distorted_timesteps = DistortTimesteps(X, sigma, knot)
    X_new = np.zeros_like(X)

    for i in range(X.shape[0]):  # For each sample
        for j in range(X.shape[2]):  # For each feature
            t_original = np.arange(X.shape[0])
            t_warped = distorted_timesteps[:, j]
            X_new[:, 0, j] = np.interp(t_original, t_warped, X[:, 0, j])
    
    # Convert the result to a PyTorch tensor
    return torch.from_numpy(X_new)

# Strongly Augmented Part
#possible_policies_s = [DA_MagWarp, DA_TimeWarp, DA_Scaling, apply_gaussian_noise]

possible_policies_s = [DA_Scaling, apply_gaussian_noise]

# Weakly Augmented Part
possible_policies_w = [DA_Scaling, apply_gaussian_noise]

def weak_augment(x, batch_size, possible_augmentations= possible_policies_w, Nw=1):
    selected_policies_w = random.sample(possible_augmentations, Nw)
    augmented_data_w = x  # start with the original data
    for policy in selected_policies_w:
        if len(augmented_data_w) == batch_size:
            augmented_data_w = policy(augmented_data_w)
    return augmented_data_w

def strong_augment(x, batch_size, possible_augmentations= possible_policies_s, Ns=2):
    selected_policies_s = random.sample(possible_augmentations, Ns)
    augmented_data_s = x  # start with the original data
    for policy in selected_policies_s:
        if len(augmented_data_s) == batch_size:
            augmented_data_s = policy(augmented_data_s)
    return augmented_data_s