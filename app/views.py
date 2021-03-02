from app import app, data_loader
from flask import render_template

@app.route("/")
def index():
  states = data_loader.DataLoader.get_states()
  daily_cases = data_loader.DataLoader.get_daily_cases_dict("MA") # TODO: update this to get nationwide results
  return render_template('index.html', state=None, states=states, daily_cases=daily_cases)

@app.route("/state/<state>")
def index_state(state):
  states = data_loader.DataLoader.get_states()
  daily_cases = data_loader.DataLoader.get_daily_cases_dict(state)
  return render_template('index.html', state=state, states=states, daily_cases=daily_cases)
