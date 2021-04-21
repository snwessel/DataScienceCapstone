from data_loader import DataLoader
from models import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import config_loader

window_size = config_loader.get_window_size()
num_days = config_loader.get_num_predicted_days()

# Load the training data with n days held out
print("Loading training data...")
X_train, y_train, y_test = DataLoader.get_date_separated_testing_data(window_size, num_days,)
# train the model (and save it to file)
print("Training the model...")
model = LinearRegression()
model.train(X_train, y_train)


# get metrics for the multi-day predictions in each state
states = DataLoader.get_states()
all_errors = np.empty((0, num_days))
all_predictions = np.empty((0, num_days))
all_actual = np.empty((0, num_days))
all_control = np.empty((0, num_days))
for state_name, state_abbrev in states.items():
  if state_abbrev != "US":
    # get the multi-day predictions 
    case_df = DataLoader.get_daily_cases_df(state_abbrev)[:-num_days]
    vax_df = DataLoader.get_daily_vaccinations_df(state_abbrev)[:-num_days]
    future_vaccinations = DataLoader.get_assumed_vaccinations_dict(vax_df, num_days, multiplier=1)
    predictions_dict = DataLoader.get_predictions(
      case_df,
      vax_df,
      future_vaccinations["vaccinations"], 
      window_size, 
      num_days, 
      model=model)
    predictions = predictions_dict["predictions"]

    # get the actual case counts 
    cases_df = DataLoader.get_daily_cases_df(state_abbrev)[-num_days:] # get the last n cases
    y_test = cases_df["new_case"].tolist()

    # compare the predicted to the actual
    errors = []
    for day in range(num_days):
      error = predictions[day] - y_test[day]
      errors.append(error)
    all_errors = np.append(all_errors, np.array([errors]), axis=0)
    all_predictions = np.append(all_predictions, np.array([predictions]), axis=0)
    all_actual = np.append(all_actual, np.array([y_test]), axis=0)

    # store the last known day's value to use to compare the control
    last_case_count = case_df["new_case"].tolist()[-1]
    all_control = np.append(all_control, [np.full((21,), last_case_count)], axis=0)
    # mse = mean_squared_error(y_test, predictions)
    # print("Errors in", state_abbrev, "are:", errors)
    # print("MSE in", state_name, "is:", mse)
    # print("R^2 in", state_name, "is", r2_score(y_test, predictions))

# double check that values corroborate each other 
calculated_errors = all_predictions - all_actual
print("Double checking calculations...")
print("\tExpected and actual shape:", all_errors.shape, calculated_errors.shape)
print("\tExpected and actual sum:", np.sum(all_errors), np.sum(calculated_errors))
print("\tall_control size is:", all_control.shape)

all_errors = np.absolute(all_errors)
daily_avgs = np.average(all_errors, axis=0)
print("Daily values averaged across states:", daily_avgs)

# get organized values for our final paper
print("Getting easily readable metrics:")
for i in range(num_days):
  if i % 5 == 0: 
    actual_vals = all_actual[:,i]
    predicted_vals = all_predictions[:,i]
    control_vals = all_control[:,i]
    r2 = r2_score(actual_vals, predicted_vals)
    mse = mean_squared_error(actual_vals, predicted_vals)
    print("\tDay", i+1, "MSE is", int(mse), "R2 is", r2)
    print("\t\tControl MSE is", int(mean_squared_error(actual_vals, control_vals)), "R2 is",
      r2_score(actual_vals, control_vals))