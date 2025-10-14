# train or inference

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from carPredictor import CarPredictor
from torch.utils.tensorboard import SummaryWriter
from utils import loadConfig, getData, denormalize, angleToDegrees

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
    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Training new model {config['name']}.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    log_dir = '../logs/' + config['name']
    writer = SummaryWriter(log_dir=log_dir)
    num_epochs = config['model']['epochs']
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

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

        # denormalize outputs and targets, only x and z (0, 2)
        outputs = denormalize(outputs, config)
        targets = denormalize(targets, config)

        # change sin, cos back to angle in degrees
        outputs, targets = angleToDegrees(outputs, targets)

        print("outputs", *outputs[0:5].cpu().numpy())
        print("targets", *targets[0:5].cpu().numpy())

    # plot all in one figure
    plt.figure(figsize=(15, 10))
    for i in range(outputs.shape[1]):
        plt.subplot(4, 4, i+1)
        # plt.plot(outputs[:, i].cpu().numpy(), label='Predicted')
        # plt.plot(targets[:, i].cpu().numpy(), label='Actual')
        # difference = abs(outputs[:, i] - targets[:, i]).cpu().numpy()

        # deal with difference for angle, handle wrap around 360
        if i == 3: # angle column
            difference = (outputs[:, i] - targets[:, i]).cpu().numpy()
            difference = (difference + 180) % 360 - 180  # wrap around
            difference = abs(difference)
        else:
            difference = abs(outputs[:, i] - targets[:, i]).cpu().numpy()

        plt.plot(difference, label='Difference', color='orange')
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