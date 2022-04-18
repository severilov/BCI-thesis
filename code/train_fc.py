import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

from models.fc_nn import BaselineNN
from dataset.base_dataset import SimulateDataset
from visualization import plot_loss

TRAIN_DATASET_PATH = "../data/train_data.pickle"
TEST_DATASET_PATH = "../data/test_data.pickle"

def predict_fc(model, x1, gen_len):
    model.eval()
    outputs = []
    inputs = x1
    for i in tqdm(range(gen_len)):
        output = np.array(model(torch.from_numpy(inputs)).detach())
        outputs.append(output)
        inputs = output
    return np.array(outputs).squeeze()

def train(trainset, testset, num_epochs=5, lr=0.01, batch_size=32, log_dir='./logs'):
    run_name = datetime.now().strftime("%d_%m_%Y.%H_%M")

    model = BaselineNN()
    trainDataLoader = DataLoader(trainset, batch_size=batch_size)
    testDataLoader = DataLoader(testset, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, valid_losses = [], []
    for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
        train_loss = 0.0
        num_batches = len(trainDataLoader)
        
        for i, data in enumerate(trainDataLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        train_losses.append(train_loss / num_batches)
        train_loss = 0.0
    
        if log_dir is not None:
            plot_loss(train_losses, valid_losses, model_name='fc', log_dir=log_dir)
            torch.save(model.state_dict(), os.path.join(log_dir, f"fc_{run_name}.pth"))


def main():
    with open(TRAIN_DATASET_PATH, 'rb') as f:
        train_data = pickle.load(f)
    with open(TEST_DATASET_PATH, 'rb') as f:
        test_data = pickle.load(f)
    
    [x_train, xt_train, y_train] = train_data
    [x_test, xt_test, y_test] = test_data
    trainset = SimulateDataset(x_train, y_train)
    testset = SimulateDataset(x_test, y_test)
    train(trainset, testset, num_epochs=5, lr=0.01, batch_size=32, log_dir='./logs')


if __name__ == '__main__':
    main()