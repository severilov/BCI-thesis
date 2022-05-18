from tqdm import tqdm
import numpy as np
import torch
import jax
import pickle

from models.lstm import LSTMModel
from models.fc_nn import BaselineNN
from dataset.make_data import solve_analytical
from visualization import plot_predicted_trajectory, radial2cartesian
from train_fc import predict_fc
from train_lstm import predict_lstm
from train_lnn import predict_lnn
from train_lnn_noether import predict_lnn as predict_lnn_new


def get_dynamics_coords(x1, model_name):
    L1, L2 = 1, 1
    theta1ana, theta2ana = x1[:, 0], x1[:, 1]
    cart_coords = radial2cartesian(theta1ana, theta2ana, L1, L2)

    print('making plots')
    plot_predicted_trajectory(cart_coords, model_name=model_name)
    return cart_coords

def main(model_name="LSTM"):
    # choose an initial state
    print('Initialization')
    x = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32) #x1
    t = np.linspace(0, 20, num=301) # num=305??? # t2
    x1_analytical = jax.device_get(solve_analytical(x, t))

    #cart_coords_ana = get_dynamics_coords(x1_analytical, model_name="Analytical solution")

    if model_name == 'LSTM':
        model = LSTMModel()
        model.load_state_dict(torch.load("./logs/lstm_18_04_2022.14_22.pth"))
        model.eval()

        # next lines ??????
        # test_list=[]
        # test_list.append(x_analytical)
        # test_set=[]

        # for seq in test_list:
        #     x = seq[[0,1,2,3],:]
        #     y = seq[4:]
        #     test_set.append((x,y))

        # output = predict_lstm(model, test_set[0][0], 301)
        
        x1_model = predict_lstm(model, x1_analytical[[0,1,2,3],:], 301)

        seg = x1_analytical[:4]
        x1_model = np.insert(x1_model, 0, seg, axis = 0)
    elif model_name == 'FC':
        model = BaselineNN()
        model.load_state_dict(torch.load("./logs/fc_18_04_2022.14_21.pth"))
        model.eval()
        x1_model = predict_fc(model, x, 301)
    elif model_name == "LNN":
        print('Making prediction')
        #model_name = 'lnn_model_15_05_2022.13_06.pickle' #best
        #model_filename = 'lnn_model_17_05_2022.19_53.pickle'
        #model_filename = 'lnn_model_18_05_2022.14_10.pickle'
        model_filename = 'lnn_model_18_05_2022.14_18.pickle'
        with open(f'./logs/{model_filename}', 'rb') as f:
            model = pickle.load(f)
        x1_model = predict_lnn(model, x, t=t)
    elif model_name == "new_LNN":
        print('Making prediction')
        #model_filename = 'new_lnn_model_17_05_2022.16_45.pickle'
        model_filename = 'new_lnn_model_18_05_2022.14_52.pickle'
        with open(f'./logs/{model_filename}', 'rb') as f:
            model = pickle.load(f)
        x1_model = predict_lnn_new(model, x, t=t)
    else:
        print(f'No realisation for {model_name}')

    cart_coords_mod = get_dynamics_coords(x1_model, model_name=model_name)


if __name__ == '__main__':
    main(model_name="new_LNN")