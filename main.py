# train or inference

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    data_list = torch.tensor(data_list, dtype=torch.float32).numpy()
    # split into train and test
    split_idx = int(0.8 * len(data_list))
    train_data = DataLoader(data_list[:split_idx], batch_size=32, shuffle=True)
    test_data = DataLoader(data_list[split_idx:], batch_size=32, shuffle=False)

    return train_data, test_data

def trainModel():
    train_data, _ = getData()
    model = CarPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        for i, data in enumerate(train_data):
            # input is current data, target is next data
            inputs = data.unsqueeze(1).to(device)
            if i + 1 < len(train_data):
                targets = torch.tensor(train_data.dataset[i+1:i+2], dtype=torch.float32).to(device)
            else:
                targets = data.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'car_predictor.pth')
    print("Model trained and saved as car_predictor.pth")

def runInference():
    _, test_data = getData()
    model = CarPredictor()
    model.load_state_dict(torch.load('car_predictor.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    accuracy = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(test_data):
            inputs = data.unsqueeze(1).to(device)
            if i + 1 < len(test_data):
                targets = torch.tensor(test_data.dataset[i+1:i+2], dtype=torch.float32).to(device)
            else:
                targets = data.to(device)

            outputs = model(inputs)
            print(f'Input: {inputs.squeeze(1).cpu().numpy()}')
            print(f'Predicted: {outputs.cpu().numpy()}')
            print(f'Actual: {targets.cpu().numpy()}')
            for j in range(outputs.size(1)): # calculate accuracy per feature
                accuracy += (1 - abs((outputs[0][j] - targets[0][j]) / (targets[0][j] + 1e-6))).item()
            if i == 5:  # limit output for brevity
                break

        print(f'Inference accuracy: {accuracy / len(test_data.dataset):.4f}')


if __name__ == "__main__":
    mode = selectMode()
    if mode == '1':
        trainModel()
    elif mode == '2':
        runInference()
    else:
        print("Invalid mode selected. Please choose 1 or 2.")