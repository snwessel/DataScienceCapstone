from app import app, models
from flask import render_template

@app.route("/")
def index():
  states = models.DataLoader.get_states()
  daily_cases = models.DataLoader.get_daily_cases("MA") # TODO: update this to get nationwide results
  return render_template('index.html', state=None, states=states, daily_cases=daily_cases)

@app.route("/state/<state>")
def index_state(state):
  states = models.DataLoader.get_states()
  daily_cases = models.DataLoader.get_daily_cases(state)
  return render_template('index.html', state=state, states=states, daily_cases=daily_cases)
