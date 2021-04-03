import data_loader
import models
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
  def __init__(self, window_size, holdout_days=0):
    self.X_train, self.X_test, self.y_train, self.y_test = self.prep_train_test(window_size, holdout_days)

  def prep_train_test(self, window_size, holdout_days=0):
    # get windowed data (for all states)
    print("Loading training and testing data...")
    X, y = data_loader.DataLoader.get_windowed_training_data(window_size, holdout_days=holdout_days)

    X_arr = X.to_numpy().astype(int)
    y_arr = np.array(y).astype(int)

    # Keep in mind: for future, maybe only include days with vaccine info?
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.33)  # default test_size: 0.25

    return X_train, X_test, y_train, y_test


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
  model = models.LinearRegression()
  model.get_best_params(train_test_data)
  model.train(train_test_data.X_train, train_test_data.y_train, save_model=save_model)
  model.display_metrics(train_test_data, show_plot=show_plot)


def train_test_ridge_regression(train_test_data, show_plot=False, save_model=False):
  model = models.RidgeRegression()
  model.get_best_params(train_test_data)
  model.train(train_test_data.X_train, train_test_data.y_train, save_model=save_model)
  model.display_metrics(train_test_data, show_plot=show_plot)
  

def train_test_lasso(train_test_data, show_plot=False, save_model=False):
  model = models.RidgeRegression()
  model.get_best_params(train_test_data)
  model.train(train_test_data.X_train, train_test_data.y_train, save_model=save_model)
  model.display_metrics(train_test_data, show_plot=show_plot)


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
