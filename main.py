# train or inference

import csv
import torch
import torch.nn as nn
from carPredictor import CarPredictor
import matplotlib.pyplot as plt
import numpy as np
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

    # Normalize the data
    data_array = np.array(data_list)
    data_mean = data_array.mean(axis=0)
    data_std = data_array.std(axis=0)
    data_list = (data_array - data_mean) / data_std

    # save mean and std for inference
    np.save('data_mean.npy', data_mean)
    np.save('data_std.npy', data_std)

    lookback = 10
    train_data = []
    target_data = []
    for i in range(len(data_list) - lookback):
        train_data.append(data_list[i:i+lookback])
        target_data.append(data_list[i+lookback])

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
    
    writer = SummaryWriter(log_dir='logs')
    num_epochs = 100000
    for epoch in range(num_epochs):
        model.train()
        inputs = train_dataset.to(device)
        targets = train_targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if epoch > 0 and epoch % 1000 == 0:
            torch.save(model.state_dict(), f'model_norm/car_predictor_{epoch}.pth')
    
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
        outputs = model(inputs)

        print("outputs", outputs)
        print("targets", targets)
        
        accuracy = 0.0
        correct = [0] * outputs.shape[1]
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if abs(outputs[i][j] - targets[i][j]) < 0.1 * abs(targets[i][j]): # within 10% margin
                    correct[j] += 1

    accuracy = [c / len(outputs) * 100 for c in correct]
    print("Accuracy per output dimension (% within 10% margin):", [f"{a:.2f}%" for a in accuracy])

    # plot each column of outputs and targets
    # for i in range(outputs.shape[1]):
    #     plt.figure()
    #     plt.plot(outputs[:, i].cpu().numpy(), label='Predicted')
    #     plt.plot(targets[:, i].cpu().numpy(), label='Actual')
    #     plt.title(f'Output Dimension {i+1}')
    #     plt.xlabel('Sample')
    #     plt.ylabel('Value')
    #     plt.legend()
    #     plt.show()




if __name__ == "__main__":
    mode = selectMode()
    if mode == '1':
        trainModel()
    elif mode == '2':
        runInference()
    else:
        print("Invalid mode selected. Please choose 1 or 2.")