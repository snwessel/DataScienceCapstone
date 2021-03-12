from app import app, data_loader
from flask import render_template

@app.route("/")
def index():
  states = data_loader.DataLoader.get_states()
  daily_cases_dict = data_loader.DataLoader.get_national_cases_dict() # TODO: see if this is correct?
  daily_vaccinations_dict = data_loader.DataLoader.get_daily_vaccinations_dict("US")
  return render_template('index.html', state=None, states=states, daily_cases=daily_cases_dict, daily_vaccinations=daily_vaccinations_dict)

@app.route("/state/<state>")
def index_state(state):
  states = data_loader.DataLoader.get_states()
  if state == "US":
    daily_cases = data_loader.DataLoader.get_national_cases_dict()
  else:
    daily_cases_df = data_loader.DataLoader.get_daily_cases_df(state)
    daily_cases_dict = data_loader.DataLoader.get_daily_cases_dict(daily_cases_df, state)
  daily_vaccinations_df = data_loader.DataLoader.get_daily_vaccinations_df(state)
  daily_vaccinations_dict = data_loader.DataLoader.get_daily_vaccinations_dict(daily_vaccinations_df, state)
  future_vaccinations = data_loader.DataLoader.get_assumed_vaccinations_dict(daily_vaccinations_df)
  return render_template('index.html', 
                          state=state, 
                          states=states, 
                          daily_cases=daily_cases_dict, 
                          daily_vaccinations=daily_vaccinations_dict,
                          future_vaccinations=future_vaccinations)
