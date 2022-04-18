from tqdm import tqdm
import numpy as np
import torch
import jax
import pickle

from models.lstm import LSTMModel
from models.fc_nn import BaselineNN
from dataset.make_data import solve_analytical
from .visualization import cart_coords_over_time, radial2cartesian
from .train_fc import predict_fc
from .train_lstm import predict_lstm
from .train_lnn import predict_lnn


def main(model_name="LSTM"):
    # choose an initial state
    x = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32) #x1
    t = np.linspace(0, 20, num=301) # num=305??? # t2
    x1_analytical = jax.device_get(solve_analytical(x, t))

    if model_name == 'LSTM':
        model = LSTMModel()
        model.load_state_dict(torch.load("LSTM_simulate"))
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
        model.load_state_dict(torch.load("FC"))
        model.eval()
        x1_model = predict_fc(model, x, 301)
    elif model_name == "LNN":
        params = pickle.load('lnn_params.pickle')
        model = params
        x1_model = predict_lnn(model, x, t=t)
    else:
        print(f'No realisation for {model_name}')

    
    L1, L2 = 1, 1
    theta1mod, theta2mod = x1_model[:, 0], x1_model[:, 1]
    cart_coords_mod = radial2cartesian(theta1mod, theta2mod, L1, L2)

    L1, L2 = 1, 1
    theta1ana, theta2ana = x1_analytical[:, 0], x1_analytical[:, 1]
    cart_coords_ana = radial2cartesian(theta1ana, theta2ana, L1, L2)

    cart_coords_over_time(cart_coords_mod)
    cart_coords_over_time(cart_coords_ana)



if __name__ == '__main__':
    main(model_name="LSTM")