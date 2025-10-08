import os
import csv
import json
import torch
import numpy as np

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def getData(config):
    data = csv.reader(open(config['data'], 'r'))
    next(data)  # skip header
    data_list = []
    for row in data: # skip first column (timestamp)
        data_list.append([float(i) for i in row[1:]])
        # replace angle with sin and cos
        angle = data_list[-1][7]
        data_list[-1][7] = np.sin(np.radians(angle))
        data_list[-1].append(np.cos(np.radians(angle)))

    print("data example:", data_list[0:5])
    # Normalize only column 0-4 and 6 (v1-v4, x, z)
    data_array = np.array(data_list)
    data_mean = np.mean(data_array[:, [0,1,2,3,4,6]], axis=0)
    data_std = np.std(data_array[:, [0,1,2,3,4,6]], axis=0)
    data_array[:, [0,1,2,3,4,6]] = (data_array[:, [0,1,2,3,4,6]] - data_mean) / data_std
    data_list = data_array.tolist()

    print("normalized data example:", data_list[0:5])
    # save mean and std for inference
    norm_dir = os.path.join(os.path.dirname(__file__), '..', config['norm_params']['directory'])
    np.save(norm_dir + '/' + config['norm_params']['mean'], data_mean)
    np.save(norm_dir + '/' + config['norm_params']['std'], data_std)

    lookback = 15
    train_data = []
    target_data = []
    for i in range(len(data_list) - lookback):
        train_data.append(data_list[i:i+lookback])
        target_data.append(data_list[i+lookback][4:9]) # xyz, sin(angle), cos(angle)

    print("target_data example:", target_data[0:5])
    split_idx = int(0.8 * len(train_data))
    train_dataset = torch.tensor(train_data[:split_idx], dtype=torch.float32)
    test_dataset = torch.tensor(train_data[split_idx:], dtype=torch.float32)
    split_idx = int(0.8 * len(target_data))
    train_targets = torch.tensor(target_data[:split_idx], dtype=torch.float32)
    test_targets = torch.tensor(target_data[split_idx:], dtype=torch.float32)

    return train_dataset, train_targets, test_dataset, test_targets