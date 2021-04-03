import data_loader

# app.config.from_pyfile('custom_configs.py')

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

# hide annoying sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class TrainTestData:
  def __init__(self, window_size):
    self.X_train, self.X_test, self.y_train, self.y_test = self.prep_train_test(window_size)

  def prep_train_test(self, window_size):
    # get windowed data (for all states)
    X, y = data_loader.DataLoader.get_windowed_training_data(window_size)

    X_arr = X.to_numpy().astype(int)
    y_arr = np.array(y).astype(int)

    # Keep in mind: for future, maybe only include days with vaccine info?
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.33)  # default test_size: 0.25

    return X_train, X_test, y_train, y_test


def display_metrics(trained_model, train_test_data, model_name="", show_plot=False):
  print(model_name, "performance:")
  y_train_pred = trained_model.predict(train_test_data.X_train)
  y_test_pred = trained_model.predict(train_test_data.X_test)

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
    plt.title(model_name + 'Residuals plot to assess heteroscedasticity')
    plt.xlabel('y_test_pred')
    plt.ylabel('residuals (y_test - y_test_pred)')
    plt.show()

def save_trained_model(model):
  s = pickle.dumps(model)
  pickle.dump(s, open("data/trained_model.p", "wb"))

def train_test_linear_regression(train_test_data, show_plot=False, save_model=False):
  # Commented out, was doing this to compare different methods of cross validation
  # # can use cross_validate instead of cross_val_score to evaluate more metrics than just R^2
  # scoring = ["r2", "neg_mean_squared_error", "neg_root_mean_squared_error"]

  # # 1) time series split
  # tscv = TimeSeriesSplit(n_splits=5)
  # print(tscv)
  # # Below shows the timeseries splits if we want to see them
  # for train, test in tscv.split(X_train):
  #   print("%s %s" % (train, test))
  # ts_scores = cross_validate(LinearRegression(), X_train, y_train, cv=tscv, scoring=scoring)
  # print(ts_scores)
  # print("Loss: {0:.3f} (+/- {1:.3f})".format(ts_scores["test_r2"].mean(), ts_scores["test_r2"].std()))
  # print("RMSE: {0:.3f} (+/- {1:.3f})".format(ts_scores["test_neg_root_mean_squared_error"].mean(), ts_scores["test_neg_root_mean_squared_error"].std()))
  # print("MIn RMSE: {0:.3f}".format(min(ts_scores["test_neg_root_mean_squared_error"])))

  # # 2) K-fold
  # kfcv = KFold(n_splits=5)
  # print(kfcv)
  # kf_scores = cross_validate(LinearRegression(), X_train, y_train, cv=kfcv, scoring=scoring)
  # print(kf_scores)
  # print("Loss: {0:.3f} (+/- {1:.3f})".format(kf_scores["test_r2"].mean(), kf_scores["test_r2"].std()))
  # print("RMSE: {0:.3f} (+/- {1:.3f})".format(kf_scores["test_neg_root_mean_squared_error"].mean(), kf_scores["test_neg_root_mean_squared_error"].std()))
  # print("Min RMSE: {0:.3f}".format(min(kf_scores["test_neg_root_mean_squared_error"])))

  # Current params to tune: {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}
  param_grid = {'fit_intercept':[True, False], 'normalize':[True, False], 'copy_X':[True, False]}

  # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
  grid = GridSearchCV(LinearRegression(), param_grid, cv=TimeSeriesSplit())
  search_results = grid.fit(train_test_data.X_train, train_test_data.y_train)
  best_params = search_results.best_params_

  # Train model with the best parameters
  # NOTE: one example had it with fit_int = False and norm = True, but if fit_int is False the norm value is ignored according to sklearn
      # param_grid can be updated to maybe exclude fit_int = False then?
  lin_reg = LinearRegression(
    copy_X=best_params["copy_X"], 
    fit_intercept=best_params["fit_intercept"], 
    normalize=best_params["normalize"]).fit(train_test_data.X_train, train_test_data.y_train)

  # save the model
  if save_model:
    print("Saving linear regression model")
    save_trained_model(lin_reg)

  # Print out performance metrics
  display_metrics(lin_reg, train_test_data, model_name="Linear Regression", show_plot=show_plot)
  

def train_test_ridge_regression(train_test_data, show_plot=False, save_model=False):

  # Current params to tune: {'alpha': 1.0, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'normalize': False, 'random_state': None, 'solver': 'auto', 'tol': 0.001}
  param_grid = {
    'alpha': [ 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

  # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
  grid = GridSearchCV(Ridge(), param_grid, cv=TimeSeriesSplit())
  search_results = grid.fit(train_test_data.X_train, train_test_data.y_train)
  best_params = search_results.best_params_

  # Train model with the best parameters
  # NOTE: just doing alpha for now, and going with the rest of the defaults and ones chosen? ran into issues trying to tune solver...
  ridge_reg = Ridge(
    alpha=best_params["alpha"], 
    solver=best_params["solver"]
    ).fit(train_test_data.X_train, train_test_data.y_train)

  # save the model
  if save_model:
    save_trained_model(ridge_reg)

  # Print out performance metrics
  display_metrics(ridge_reg, train_test_data, model_name="Ridge Regression", show_plot=show_plot)


def train_test_lasso(train_test_data, show_plot=False, save_model=False):
  
  # Current params to tune: {'alpha': 1.0, 'copy_X': True, 'fit_intercept': True, 'max_iter': 1000, 'normalize': False, 'positive': False, 'precompute': False, 'random_state': None, 'selection': 'cyclic', 'tol': 0.0001, 'warm_start': False}
  param_grid = {
    'alpha': [ 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0], 
    'tol': [0.0001, 0.001, 0.01, 0.1]
    }

  # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
  grid = GridSearchCV(Lasso(), param_grid, cv=TimeSeriesSplit())
  search_results = grid.fit(train_test_data.X_train, train_test_data.y_train)
  best_params = search_results.best_params_

  # Train model with the best parameters
  lasso_reg = Lasso(
    alpha=best_params["alpha"], 
    tol=best_params["tol"]
    ).fit(train_test_data.X_train, train_test_data.y_train)

  # save the model
  if save_model:
    save_trained_model(lasso_reg)

  # Print out performance metrics
  display_metrics(lasso_reg, train_test_data, model_name="Lasso Regression", show_plot=show_plot)


def train_test_control(windowed_data):
  print("Control Performance:")
  y_train_pred = windowed_data.X_train[:, window_size-2]
  y_test_pred = windowed_data.X_test[:, window_size-2]
  print('\tMSE train: %.3f, test: %.3f' % (mean_squared_error(windowed_data.y_train, y_train_pred),
                  mean_squared_error(windowed_data.y_test, y_test_pred)))
  print('\tR^2 train: %.3f, test: %.3f' % (r2_score(windowed_data.y_train, y_train_pred),
                  r2_score(windowed_data.y_test, y_test_pred)))


### Performance Analysis ###
display_graphs = False
window_sizes = [8, 12, 16, 24, 32]

# get configured window size (so we know which one to save)
window_size = 24 #app.config['WINDOW_SIZE']

for window_size in window_sizes:
  print("\nEvaluating models on window_size", window_size, "\n----------")
  windowed_data = TrainTestData(window_size)

  save_lin_reg = (window_size == 24) # save lin reg model when window=24

  train_test_linear_regression(windowed_data, display_graphs, save_model=save_lin_reg)
  train_test_ridge_regression(windowed_data, display_graphs)
  train_test_lasso(windowed_data, display_graphs)
  train_test_control(windowed_data)
