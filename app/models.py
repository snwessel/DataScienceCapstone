import torch
import torch.nn as nn
from torch.autograd import Variable 

# An interface for interacting with machine learning models
class MLModel:
  def fit(self, daily_cases, daily_vaccinations):
    """Train the model to fit the data"""
    pass

  def predict(self, num_days):
    """Predict the given number of days"""
    # We may need to pass in the state data too
    pass 

class LSTM(nn.Module):
  def __init__(self):
    self.input_dim = 2
    self.hidden_size = 20
    self.n_layers = 1
    self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.n_layers)
    # TODO: add a linear layer at the end here??

  def forward(self, x):
    """Train the model to fit the data"""
    h_0 = Variable(torch.zeros(
          self.n_layers, x.size(0), self.hidden_size)) #hidden state
    c_0 = Variable(torch.zeros(
        self.n_layers, x.size(0), self.hidden_size)) #internal state
   
    # Propagate input through LSTM
    output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
    # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
    # out = self.relu(hn)
    # out = self.fc_1(out) #first Dense
    # out = self.relu(out) #relu
    # out = self.fc(out) #Final Output
    return output 

  def backward(self, outputs, y_train_tensors, learning_rate):
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(self.lstm.parameters(), lr=learning_rate) 
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
  
    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)
  
    loss.backward() #calculates the loss of the loss function
    optimizer.step() #improve from loss, i.e backprop
    return loss



def train_LSTM(X_train_tensors, y_train_tensors, learning_rate):
  lstm = LSTM()

  for epoch in range(100):
    outputs = lstm.forward(X_train_tensors) #forward pass
    loss = lstm.backward(outputs, y_train_tensors, learning_rate) # backward pass

    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) # logging




