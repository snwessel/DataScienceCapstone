from data_loader import DataLoader
from models import LinearRegression
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
model.train(X_train, y_train, save_model=True)


# get metrics for the multi-day predictions in each state
states = DataLoader.get_states()
for state_name, state_abbrev in states.items():
  # get the multi-day predictions 
  case_df = DataLoader.get_daily_cases_df(state_abbrev)[:-num_days]
  vax_df = DataLoader.get_daily_vaccinations_df(state_abbrev)[:-num_days]
  future_vaccinations = DataLoader.get_assumed_vaccinations_dict(vax_df, num_days, multiplier=1)
  predictions_dict = DataLoader.get_predictions(
    case_df,
    vax_df,
    future_vaccinations["vaccinations"], 
    window_size, 
    num_days)
  predictions = predictions_dict["predictions"]

  # get the actual case counts 
  cases_df = DataLoader.get_daily_cases_df(state_abbrev)[-num_days:] # get the last n cases
  y_test = cases_df["new_case"].tolist()

  # compare the predicted to the actual
  errors = []
  for day in range(num_days):
    error = predictions[day] - y_test[day]
    errors.append(int(error))
  mse = mean_squared_error(y_test, predictions)
  #print("Errors in", state_abbrev, "are:", errors)
  print("MSE in", state_name, "is:", mse)
  print("R^2 in", state_name, "is", r2_score(y_test, predictions))