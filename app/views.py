from app import app, models
from flask import render_template

@app.route("/")
def index():
  states = models.DataLoader.get_states()
  return render_template('index.html', state=None, states=states)

@app.route("/state/<state>")
def index_state(state):
  states = models.DataLoader.get_states()
  return render_template('index.html', state=state, states=states)
