from app import app, data_loader
from flask import render_template, request

@app.route("/")
def index():
  # get configs 
  window_size = app.config['WINDOW_SIZE'] # default to 7 if not found

  args = request.args.to_dict()
  state = args.get("state", "MA") # temporarily defaulting to MA since vaccinations aren't loading for the US yet
  multiplier = float(args.get("multiplier", "1"))

  all_states = data_loader.DataLoader.get_states()
  # get daily case counts
  if state == "US":
    daily_cases_dict = data_loader.DataLoader.get_national_cases_dict()
  else:
    daily_cases_df = data_loader.DataLoader.get_daily_cases_df(state)
    daily_cases_dict = data_loader.DataLoader.get_daily_cases_dict(daily_cases_df, state)
  # get total vaccination counts
  daily_vaccinations_df = data_loader.DataLoader.get_daily_vaccinations_df(state)
  daily_vaccinations_dict = data_loader.DataLoader.get_daily_vaccinations_dict(daily_vaccinations_df, state)
  future_vaccinations = data_loader.DataLoader.get_assumed_vaccinations_dict(daily_vaccinations_df, multiplier=multiplier)
  # get exploratory data
  state_population_dict = data_loader.DataLoader.get_state_population_counts_dict()
  influenza_dict = data_loader.DataLoader.get_infuenza_counts_dict()
  # get predictions
  predictions = data_loader.DataLoader.get_predictions(daily_cases_df, daily_vaccinations_df, future_vaccinations["vaccinations"], window_size, 30)
  
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
                          predictions=predictions)

