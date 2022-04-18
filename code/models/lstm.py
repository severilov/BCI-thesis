from torch import nn

class LSTMModel(nn.Module):
    def __init__(self):
        # We want a model of 4 layer LSTM with 32 features output, and a dense layer to form the 4 feature output.
        super(LSTMModel, self).__init__()

        # Defining some parameters
        #self.hidden_size = 32
        #self.n_layers = 4

        #Defining the layers
        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(8, 4)
    
    def forward(self, x):
        out1, _= self.lstm1(x) # (h0.detach(), c0.detach())
        out2, _= self.lstm2(out1)
        out3, _= self.lstm3(out2)
        out3 = out3[:, -1, :]
        out = self.fc(out3)
        return out


def init_lstm():
    pass