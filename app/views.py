from app import app, data_loader
from flask import render_template

@app.route("/")
def index():
  states = data_loader.DataLoader.get_states()
  daily_cases = data_loader.DataLoader.get_national_cases_dict() # TODO: see if this is correct?
  daily_vaccinations = data_loader.DataLoader.get_daily_vaccinations_dict("US")
  return render_template('index.html', state=None, states=states, daily_cases=daily_cases, daily_vaccinations=daily_vaccinations)

@app.route("/state/<state>")
def index_state(state):
  states = data_loader.DataLoader.get_states()
  if state == "US":
    daily_cases = data_loader.DataLoader.get_national_cases_dict()
  else:
    daily_cases = data_loader.DataLoader.get_daily_cases_dict(state)
  daily_vaccinations = data_loader.DataLoader.get_daily_vaccinations_dict(state)
  return render_template('index.html', state=state, states=states, daily_cases=daily_cases, daily_vaccinations=daily_vaccinations)
