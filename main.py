# train or inference

import csv
import torch
import torch.nn as nn
from carPredictor import CarPredictor

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def getData():
    data = csv.reader(open('carData.csv', 'r'))
    next(data)  # skip header
    data_list = []
    for row in data: # skip first column (timestamp)
        data_list.append([float(i) for i in row[1:]])

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
    
    num_epochs = 2000
    for epoch in range(num_epochs):
        model.train()
        inputs = train_dataset.to(device)
        targets = train_targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'car_predictor.pth')
    print("Model trained and saved as car_predictor.pth")

def runInference():
    _, _, test_dataset, test_targets = getData()
    model = CarPredictor()
    model.load_state_dict(torch.load('car_predictor.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    accuracy = 0.0
    
    with torch.no_grad():
        inputs = test_dataset.to(device)
        targets = test_targets.to(device)
        outputs = model(inputs)

        print("outputs", outputs)
        print("targets", targets)
        
        for i in range(len(outputs)):
            pred = outputs[i].cpu().numpy()
            actual = targets[i].cpu().numpy()
            correct = sum(1 for p, a in zip(pred, actual) if abs(p - a) < 0.1)
            accuracy += correct / len(actual)

    accuracy /= len(outputs)
    print(f'Inference completed. Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    mode = selectMode()
    if mode == '1':
        trainModel()
    elif mode == '2':
        runInference()
    else:
        print("Invalid mode selected. Please choose 1 or 2.")