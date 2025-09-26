# train or inference

import csv
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from carPredictor import CarPredictor
from torch.utils.tensorboard import SummaryWriter

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def getData():
    data = csv.reader(open('carData_1.csv', 'r'))
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
    np.save('data_mean.npy', data_mean)
    np.save('data_std.npy', data_std)

    lookback = 10
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

def trainModel():
    train_dataset, train_targets, _, _ = getData()
    model = CarPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    writer = SummaryWriter(log_dir='logs/model_params_new')
    num_epochs = 100000
    for epoch in range(num_epochs):
        model.train()
        inputs = train_dataset.to(device)
        targets = train_targets.to(device)
        predicts = model(inputs)
        
        # predicts delta, add each predict to each input's last frame 4:9 (x, y, z, sin(angle), cos(angle))
        outputs = inputs[:, -1, 4:9] + predicts

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if epoch > 0 and epoch % 100 == 0:
            torch.save(model.state_dict(), f'model_params_new/car_predictor_{epoch}.pth')
    
    writer.close()
    torch.save(model.state_dict(), 'car_predictor.pth')
    print("Model trained and saved as car_predictor.pth")

def runInference():
    _, _, test_dataset, test_targets = getData()
    model = CarPredictor()
    model.load_state_dict(torch.load('car_predictor.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        inputs = test_dataset.to(device)
        targets = test_targets.to(device)
        predicts = model(inputs)

        # predicts delta, add each predict to each input's last frame 4:9 (x, y, z, sin(angle), cos(angle))
        outputs = inputs[:, -1, 4:9] + predicts

        print("outputs", *outputs[0:5].cpu().numpy())
        print("targets", *targets[0:5].cpu().numpy())

        accuracy = 0.0
        correct = [0] * outputs.shape[1]
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if abs(outputs[i][j] - targets[i][j]) < 0.1 * abs(targets[i][j]): # within 10% margin
                    correct[j] += 1

    accuracy = [c / len(outputs) * 100 for c in correct]
    print("Accuracy per output dimension (% within 10% margin):", [f"{a:.2f}%" for a in accuracy])

    # plot all in one figure
    plt.figure(figsize=(15, 10))
    for i in range(outputs.shape[1]):
        plt.subplot(4, 4, i+1)
        plt.plot(outputs[:, i].cpu().numpy(), label='Predicted')
        plt.plot(targets[:, i].cpu().numpy(), label='Actual')
        plt.title(f'dim {i+1}')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.savefig('result_new.png')


if __name__ == "__main__":
    mode = selectMode()
    if mode == '1':
        trainModel()
    elif mode == '2':
        runInference()
    else:
        print("Invalid mode selected. Please choose 1 or 2.")