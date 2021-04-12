import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import time
import torch
import torch.nn as nn
from torch.autograd import Variable 


def save_trained_model(model):
  s = pickle.dumps(model)
  pickle.dump(s, open("data/trained_model.p", "wb"))

# an abstract class which has functions that are helpful for all of the regression models
class WindowBasedModel():
  """An interface for the regression models."""
  def __init__(self):
    self.model = None
    self.model_name = ""

  def get_best_params(self, train_test_data):
    """Perform cross validation to get the best parameters."""
    pass

  def train(self, X_train, y_train, save_model=False):
    pass

  def predict(self, x):
    return self.model.predict(x)

  def display_metrics(self, train_test_data, show_plot=False, selected_params=None):
    print(self.model_name, "performance:")
    y_train_pred = self.model.predict(train_test_data.X_train)
    y_test_pred = self.model.predict(train_test_data.X_test)

    if selected_params:
      print("\tSelected parameters:", selected_params)

    # Mean Squared Error
    print('\tMSE train: %.3f, test: %.3f' % (mean_squared_error(train_test_data.y_train, y_train_pred),
                    mean_squared_error(train_test_data.y_test, y_test_pred)))
    # R-Squared
    print('\tR^2 train: %.3f, test: %.3f' % (r2_score(train_test_data.y_train, y_train_pred),
                    r2_score(train_test_data.y_test, y_test_pred)))

    # Checking if linear regression was the best idea (plotting residuals)
    if show_plot:
      residuals = (train_test_data.y_test - y_test_pred)
      plt.scatter(y_test_pred, residuals)
      plt.title(self.model_name + 'Residuals plot to assess heteroscedasticity')
      plt.xlabel('y_test_pred')
      plt.ylabel('residuals (y_test - y_test_pred)')
      plt.show()


class MultiLayerPerceptron(WindowBasedModel):
  def __init__(self):
    """Initialize the hyperparameters to what we have found to perform the best in the past."""
    self.best_params = {"activation": "relu", "solver": "lbfgs", "hidden_layer_sizes": (100,)}
    self.model = None
    self.model_name = "Multi-Layer Perceptron"

  def get_best_params(self, train_test_data):
    """Perform cross validation to get the best parameters."""
    print("Finding the best parameters for the MLP")
    start_time = time.perf_counter()
    param_grid = {
      "activation": ["relu"], #["logistic", "tanh", "relu"], 
      "solver": ["lbfgs"], # ["lbfgs", "adam"],
      "hidden_layer_sizes": [(100,), (200,)]# [(100,), (100, 50), (150,100,50)]
      }
    grid = GridSearchCV(MLPRegressor(), param_grid, cv=TimeSeriesSplit())
    search_results = grid.fit(train_test_data.X_train, train_test_data.y_train)
    self.best_params = search_results.best_params_
    print("MLP Cross validation took", time.perf_counter() - start_time, "seconds.")
    return self.best_params

  def train(self, X_train, y_train, save_model=False):
    self.model = MLPRegressor(random_state=1, 
      max_iter=500,
      activation=self.best_params["activation"],
      solver=self.best_params["solver"],
      hidden_layer_sizes=self.best_params["hidden_layer_sizes"]
      ).fit(X_train, y_train)
    if save_model:
      print("Saving MLP model")
      save_trained_model(self.model)

class LinearRegression(WindowBasedModel):
  def __init__(self):
    """Initialize the hyperparameters to what we have found to perform the best in the past."""
    self.best_params = {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}
    self.model = None
    self.model_name = "Linear Regression"

  def get_best_params(self, train_test_data):
    """Perform cross validation to get the best parameters."""
    # Current params to tune: {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}
    param_grid = {'fit_intercept':[True, False], 'normalize':[True, False], 'copy_X':[True, False]}
    # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
    grid = GridSearchCV(linear_model.LinearRegression(), param_grid, cv=TimeSeriesSplit())
    search_results = grid.fit(train_test_data.X_train, train_test_data.y_train)
    self.best_params = search_results.best_params_
    return self.best_params

  def train(self, X_train, y_train, save_model=False):
    self.model = linear_model.LinearRegression(
      copy_X=self.best_params["copy_X"], 
      fit_intercept=self.best_params["fit_intercept"], 
      normalize=self.best_params["normalize"]).fit(X_train, y_train)
    if save_model:
      print("Saving linear regression model")
      save_trained_model(self.model)

class RidgeRegression(WindowBasedModel):
  def __init__(self):
    """Initialize the hyperparameters to what we have found to perform the best in the past."""
    self.best_params = {
      'alpha': 1.0, 
      'solver': 'auto'
    }
    self.model = None
    self.model_name = "Ridge Regression"

  def get_best_params(self, train_test_data):
    """Perform cross validation to get the best parameters."""
    param_grid = {
      'alpha': [ 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0],
      'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
      }
    # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
    grid = GridSearchCV(linear_model.Ridge(), param_grid, cv=TimeSeriesSplit())
    search_results = grid.fit(train_test_data.X_train, train_test_data.y_train)
    self.best_params = search_results.best_params_
    return self.best_params
  
  def train(self, X_train, y_train, save_model=False):
    # Train model with the best parameters
    # NOTE: just doing alpha for now, and going with the rest of the defaults and ones chosen? ran into issues trying to tune solver...
    self.model = linear_model.Ridge(
      alpha=self.best_params["alpha"], 
      solver=self.best_params["solver"]
      ).fit(X_train, y_train)
    if save_model:
      print("Saving ridge regression model")
      save_trained_model(self.model)

class LassoRegression(WindowBasedModel):
  def __init__(self):
    """Initialize the hyperparameters to what we have found to perform the best in the past."""
    self.best_params = {'alpha': 1.0, 'tol': 0.0001}
    self.model = None
    self.model_name = "Lasso Regression"

  def get_best_params(self, train_test_data):
    """Perform cross validation to get the best parameters."""
    param_grid = {
      'alpha': [ 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0], 
      'tol': [0.0001, 0.001, 0.01, 0.1]
      }

    # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
    grid = GridSearchCV(linear_model.Lasso(), param_grid, cv=TimeSeriesSplit())
    search_results = grid.fit(train_test_data.X_train, train_test_data.y_train)
    self.best_params = search_results.best_params_
    return self.best_params
  
  def train(self, X_train, y_train, save_model=False):
    # Train model with the best parameters
    # NOTE: just doing alpha for now, and going with the rest of the defaults and ones chosen? ran into issues trying to tune solver...
    self.model = linear_model.Lasso(
      alpha=self.best_params["alpha"], 
      tol=self.best_params["tol"]
      ).fit(X_train, y_train)
    if save_model:
      print("Saving lasso regression model")
      save_trained_model(self.model)




# Note: The LSTM is not working yet
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




