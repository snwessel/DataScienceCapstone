from app import app
from flask import render_template

@app.route("/")
def index():
  return render_template('index.html', state=None)

@app.route("/state/<state>")
def index_state(state):
  return render_template('index.html', state=state)
