import data_loader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_test_linear_regression():
  window_size = 7
  # get windowed data (for all states)
  X, y = data_loader.DataLoader.get_windowed_training_data(window_size)

  X_arr = X.to_numpy().astype(int)
  y_arr = np.array(y).astype(int)

  # Keep in mind: for future, maybe only include days with vaccine info?
  X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.33)  # default test_size: 0.25

  lin_reg = LinearRegression().fit(X_train, y_train)

  # save the model
  s = pickle.dumps(lin_reg)
  pickle.dump(s, open("data/trained_model.p", "wb"))

  # Print out performance metrics
  print("Linear regression performance:")
  y_train_pred = lin_reg.predict(X_train)
  y_test_pred = lin_reg.predict(X_test)
  # Mean Squared Error
  print('\tMSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                  mean_squared_error(y_test, y_test_pred)))
  # R-Squared
  print('\tR^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                  r2_score(y_test, y_test_pred)))

  # # compare to a model which just assumes that the next day will be the same as the last
  # print("Dumb Model Performance:")
  # y_train_pred = X_train[:, window_size-2]
  # y_test_pred = X_test[:, window_size-2]
  # print('\tMSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
  #                 mean_squared_error(y_test, y_test_pred)))
  # print('\tR^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
  #                 r2_score(y_test, y_test_pred)))


train_test_linear_regression()