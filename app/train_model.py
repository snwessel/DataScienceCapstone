import data_loader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def prep_train_test(n_windows):
  window_size = n_windows
  # get windowed data (for all states)
  X, y = data_loader.DataLoader.get_windowed_training_data(window_size)

  X_arr = X.to_numpy().astype(int)
  y_arr = np.array(y).astype(int)

  # Keep in mind: for future, maybe only include days with vaccine info?
  X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.33)  # default test_size: 0.25

  return X_train, X_test, y_train, y_test

def train_test_linear_regression(n_windows):
  X_train, X_test, y_train, y_test = prep_train_test(n_windows)

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
  search_results = grid.fit(X_train, y_train)
  best_params = search_results.best_params_

  # Train model with the best parameters
  # NOTE: one example had it with fit_int = False and norm = True, but if fit_int is False the norm value is ignored according to sklearn
      # param_grid can be updated to maybe exclude fit_int = False then?
  lin_reg = LinearRegression(copy_X=best_params["copy_X"], fit_intercept=best_params["fit_intercept"], normalize=best_params["normalize"]).fit(X_train, y_train)

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

  # Checking if linear regression was the best idea (plotting residuals)
  residuals = (y_test - y_test_pred)
  print("residuals")
  print(residuals)
  plt.scatter(y_test_pred, residuals)
  plt.title('OLS Residuals plot to assess heteroscedasticity')
  plt.xlabel('y_test_pred')
  plt.ylabel('residuals (y_test - y_test_pred)')
  plt.show()
  

def train_test_ridge_regression(n_windows):
  X_train, X_test, y_train, y_test = prep_train_test(n_windows)

  # Current params to tune: {'alpha': 1.0, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'normalize': False, 'random_state': None, 'solver': 'auto', 'tol': 0.001}
  param_grid = {
    'alpha': [ 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

  # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
  grid = GridSearchCV(Ridge(), param_grid, cv=TimeSeriesSplit())
  search_results = grid.fit(X_train, y_train)
  best_params = search_results.best_params_

  # Train model with the best parameters
  # NOTE: just doing alpha for now, and going with the rest of the defaults and ones chosen? ran into issues trying to tune solver...
  ridge_reg = Ridge(alpha=best_params["alpha"], solver=best_params["solver"]).fit(X_train, y_train)

  # Print out performance metrics
  print("Ridge regression performance:")
  y_train_pred = ridge_reg.predict(X_train)
  y_test_pred = ridge_reg.predict(X_test)
  # Mean Squared Error
  print('\tMSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                  mean_squared_error(y_test, y_test_pred)))
  # R-Squared
  print('\tR^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                  r2_score(y_test, y_test_pred)))

  # Checking if ridge regression was the best idea (plotting residuals)
  residuals = (y_test - y_test_pred)
  print("residuals")
  print(residuals)
  plt.scatter(y_test_pred, residuals)
  plt.title('Ridge Residuals plot to assess heteroscedasticity')
  plt.xlabel('y_test_pred')
  plt.ylabel('residuals (y_test - y_test_pred)')
  plt.show()

# NOTE: getting this warning running lasso
#  ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations.
def train_test_lasso(n_windows):
  X_train, X_test, y_train, y_test = prep_train_test(n_windows)
  
  # Current params to tune: {'alpha': 1.0, 'copy_X': True, 'fit_intercept': True, 'max_iter': 1000, 'normalize': False, 'positive': False, 'precompute': False, 'random_state': None, 'selection': 'cyclic', 'tol': 0.0001, 'warm_start': False}
  param_grid = {
    'alpha': [ 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0], 
    'tol': [0.0001, 0.001, 0.01, 0.1]
    }

  # CV can be an int for number of folds (KFold) but can also be a CV-Splitter (like TimeSeriesSplit)
  grid = GridSearchCV(Lasso(), param_grid, cv=TimeSeriesSplit())
  search_results = grid.fit(X_train, y_train)
  best_params = search_results.best_params_

  # Train model with the best parameters
  lasso_reg = Lasso(alpha=best_params["alpha"], tol=best_params["tol"]).fit(X_train, y_train)

  # Print out performance metrics
  print("Lasso regression performance:")
  y_train_pred = lasso_reg.predict(X_train)
  y_test_pred = lasso_reg.predict(X_test)
  # Mean Squared Error
  print('\tMSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                  mean_squared_error(y_test, y_test_pred)))
  # R-Squared
  print('\tR^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                  r2_score(y_test, y_test_pred)))

  # Checking if ridge regression was the best idea (plotting residuals)
  residuals = (y_test - y_test_pred)
  print("residuals")
  print(residuals)
  plt.scatter(y_test_pred, residuals)
  plt.title('Lasso Residuals plot to assess heteroscedasticity')
  plt.xlabel('y_test_pred')
  plt.ylabel('residuals (y_test - y_test_pred)')
  plt.show()


def train_test_control():
  """Test the performance of a model which assumes tomorrow's cases will be the same as today's."""
  window_size = 5 # window size doesn't really affect the performance here 
  X_train, X_test, y_train, y_test = prep_train_test(window_size) 
  
  print("Control Performance:")
  y_train_pred = X_train[:, window_size-2]
  y_test_pred = X_test[:, window_size-2]
  print('\tMSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                  mean_squared_error(y_test, y_test_pred)))
  print('\tR^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                  r2_score(y_test, y_test_pred)))


### Performance Analysis ###
# train_test_linear_regression(5)
# train_test_ridge_regression(5)
# train_test_lasso(5)

# train_test_linear_regression(7)
# train_test_ridge_regression(7)
# train_test_lasso(7)

#train_test_control()
