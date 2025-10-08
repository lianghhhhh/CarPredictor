# train or inference

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from carPredictor import CarPredictor
from utils import loadConfig, getData
from torch.utils.tensorboard import SummaryWriter

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def trainModel(config):
    train_dataset, train_targets, _, _ = getData(config)
    model = CarPredictor(
        hidden_size=config['model']['hidden_size'], 
        num_layers=config['model']['num_layers'], 
        dropout=config['model']['dropout']
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    log_dir = '../logs/' + config['name']
    writer = SummaryWriter(log_dir=log_dir)
    num_epochs = 100
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

        # create dir
        os.makedirs(f'../models/{config["name"]}', exist_ok=True)
        if epoch > 0 and epoch % 100 == 0:
            torch.save(model.state_dict(), f'../models/{config["name"]}/{epoch}.pth')

    writer.close()
    torch.save(model.state_dict(), f'../{config["name"]}.pth')
    print(f"Model trained and saved as {config['name']}.pth")

def runInference(config):
    _, _, test_dataset, test_targets = getData(config)
    model = CarPredictor(
        hidden_size=config['model']['hidden_size'], 
        num_layers=config['model']['num_layers'], 
        dropout=config['model']['dropout']
    )
    model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
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
                # # change output and target back to original scale for accuracy calculation
                # data_mean = np.load('data_mean.npy')
                # data_std = np.load('data_std.npy')
                # if j == 0: # x
                #     outputs[i][j] = outputs[i][j] * data_std[4] + data_mean[4]
                #     targets[i][j] = targets[i][j] * data_std[4] + data_mean[4]
                # elif j == 2: # z
                #     outputs[i][j] = outputs[i][j] * data_std[5] + data_mean[5]
                #     targets[i][j] = targets[i][j] * data_std[5] + data_mean[5]
                # # change angle back to degrees
                # elif j == 3 or j == 4: # sin(angle), cos(angle)
                #     angle_out = np.degrees(np.arctan2(outputs[i][3].cpu().numpy(), outputs[i][4].cpu().numpy()))
                #     angle_tgt = np.degrees(np.arctan2(targets[i][3].cpu().numpy(), targets[i][4].cpu().numpy()))
                #     outputs[i][j] = angle_out.item()
                #     targets[i][j] = angle_tgt.item()

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
    fig_path = os.path.join(os.path.dirname(__file__), '..', config['result_png'])
    plt.savefig(fig_path)


if __name__ == "__main__":
    mode = selectMode()
    config = loadConfig()
    if mode == '1':
        trainModel(config)
    elif mode == '2':
        runInference(config)
    else:
        print("Invalid mode selected. Please choose 1 or 2.")