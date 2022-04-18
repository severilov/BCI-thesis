import random
import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import jax
from datetime import datetime

from torch.utils.data import DataLoader

from models.lstm import LSTMModel
from dataset.make_data import solve_analytical
from dataset.base_dataset import DatasetLSTM
from visualization import plot_loss


def predict_lstm(model, warming_up, gen_len):
    model.eval()
    outputs = []
    inputs = warming_up 
    for i in tqdm(range(gen_len)):
        output = np.array(model(torch.from_numpy(inputs)[None,:,:]).detach())
        outputs.append(output)
        inputs = np.concatenate((inputs[[1,2,3],:],output), axis = 0)
    return np.array(outputs).squeeze()

def train(dataset, 
          test_set, 
          num_epochs=10, 
          lr=0.01, 
          batch_size=256, 
          log_dir='./logs'):
    """
    """
    run_name = datetime.now().strftime("%d_%m_%Y.%H_%M")

    model = LSTMModel()
    trainDataLoader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, valid_losses, train_losses_final = [], [], []
    print('Training Start')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        num_batches = len(trainDataLoader)

        for i, data in enumerate(trainDataLoader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                #print('[%d, %5d] loss: %.6f' %
                    #(epoch + 1, i + 1, running_loss / 100))
                train_losses.append(running_loss)
                running_loss = 0.0

        train_losses_final.append(train_loss / num_batches)
        train_loss = 0.0

        #validate
        index = 1 #random.randint(1,9)
        val_output = torch.from_numpy(predict_lstm(model, test_set[index][0],301))
        target = torch.from_numpy(test_set[index][1])
        valid_losses.append(criterion(val_output,target).item())

    print('Finished Training')
    if log_dir is not None:
        plot_loss(train_losses_final, valid_losses, model_name='lstm', log_dir=log_dir)
        torch.save(model.state_dict(), os.path.join(log_dir, f"lstm_{run_name}.pth"))

def main():
    #generate 10 sequences pf length 2000
    seq_list = []
    for i in tqdm(range(100)):
        c1 = random.random()
        c2 = random.random()
        c3 = random.random()
        x = np.array([c1*np.pi, c2*np.pi, 0, 0], dtype=np.float32)
        t = np.linspace(0, 20, num=2000)
        x_analytical = jax.device_get(solve_analytical(x, t))
        seq_list.append(x_analytical)

    dataset = DatasetLSTM(seq_list)

    test_list = []
    for i in tqdm(range(10)): #generate 10 sequences pf length 2000
        c1 = random.random()
        c2 = random.random()
        x = np.array([c1*np.pi, c2*np.pi, 0, 0], dtype=np.float32)
        t = np.linspace(0, 20, num=305)
        x_analytical = jax.device_get(solve_analytical(x, t))
        test_list.append(x_analytical)
    test_set=[]
    for seq in test_list:
        x = seq[[0,1,2,3],:]
        y = seq[4:]
        test_set.append((x,y))

    train(dataset, test_set)

if __name__ == '__main__':
    main()