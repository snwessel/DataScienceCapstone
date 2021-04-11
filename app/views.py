from app import config_loader
from app import app 
from app.data_loader import DataLoader
from flask import render_template, request

@app.route("/")
def index():
  # get configs
  window_size = config_loader.get_window_size() 
  num_to_predict = config_loader.get_num_predicted_days()

  args = request.args.to_dict()
  state = args.get("state", "MA") # temporarily defaulting to MA since vaccinations aren't loading for the US yet
  multiplier = float(args.get("multiplier", "1"))

  all_states = DataLoader.get_states()
  # get daily case counts
  if state == "US":
    daily_cases_dict = DataLoader.get_national_cases_dict()
  else:
    daily_cases_df = DataLoader.get_daily_cases_df(state)
    daily_cases_dict = DataLoader.get_daily_cases_dict(daily_cases_df, state)
  # get total vaccination counts
  daily_vaccinations_df = DataLoader.get_daily_vaccinations_df(state)
  daily_vaccinations_dict = DataLoader.get_daily_vaccinations_dict(daily_vaccinations_df, state)
  future_vaccinations = DataLoader.get_assumed_vaccinations_dict(daily_vaccinations_df, num_to_predict, multiplier=multiplier)
  # get exploratory data
  state_population_dict = DataLoader.get_state_population_counts_dict()
  influenza_dict = DataLoader.get_infuenza_counts_dict()
  national_total_vax_df = DataLoader.get_total_vaccinations_per_hundred_df()
  national_total_vax_dict = DataLoader.get_total_vaccinations_per_hundred_dict(national_total_vax_df)
  national_distrib_vax_df = DataLoader.get_total_distributed_vaccines_per_hundred_df()
  national_distrib_vax_dict = DataLoader.get_total_distributed_vaccines_per_hundred_dict(national_distrib_vax_df)

  # get predictions
  predictions = DataLoader.get_predictions(daily_cases_df, daily_vaccinations_df, future_vaccinations["vaccinations"], window_size, num_to_predict)
  
  # render the HTML template
  return render_template('index.html', 
                          state=state, 
                          multiplier=multiplier,
                          all_states=all_states, 
                          daily_cases=daily_cases_dict, 
                          daily_vaccinations=daily_vaccinations_dict,
                          future_vaccinations=future_vaccinations,
                          state_pop=state_population_dict,
                          influenza_counts=influenza_dict,
                          national_total_vax_dict=national_total_vax_dict,
                          national_distrib_vax_dict=national_distrib_vax_dict,
                          predictions=predictions)

