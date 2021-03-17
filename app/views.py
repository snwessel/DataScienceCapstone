from app import app, data_loader
from flask import render_template, request

@app.route("/")
def index():
  args = request.args.to_dict()
  state = args.get("state", "US")
  multiplier = float(args.get("multiplier", "1"))

  all_states = data_loader.DataLoader.get_states()
  if state == "US":
    daily_cases_dict = data_loader.DataLoader.get_national_cases_dict()
  else:
    daily_cases_df = data_loader.DataLoader.get_daily_cases_df(state)
    daily_cases_dict = data_loader.DataLoader.get_daily_cases_dict(daily_cases_df, state)
  daily_vaccinations_df = data_loader.DataLoader.get_daily_vaccinations_df(state)
  daily_vaccinations_dict = data_loader.DataLoader.get_daily_vaccinations_dict(daily_vaccinations_df, state)
  future_vaccinations = data_loader.DataLoader.get_assumed_vaccinations_dict(daily_vaccinations_df, multiplier=multiplier)
  state_population_dict = data_loader.DataLoader.get_state_population_counts_dict()
  influenza_dict = data_loader.DataLoader.get_infuenza_counts_dict()
  return render_template('index.html', 
                          state=state, 
                          multiplier=multiplier,
                          all_states=all_states, 
                          daily_cases=daily_cases_dict, 
                          daily_vaccinations=daily_vaccinations_dict,
                          future_vaccinations=future_vaccinations,
                          state_pop=state_population_dict,
                          influenza_counts=influenza_dict)

