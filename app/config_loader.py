import json 

# Load configs from file (so we can reference a single value from multiple scripts)

def get_num_predicted_days():
  """Get the number of days which should be predicted by our model."""
  return load_configs()["num_predicted_days"]

def get_window_size():
  """Get the number of days that should be included in the 'window' for our regression models."""
  return load_configs()["window_size"]

def load_configs():
  f = open('configs.json')
  return json.load(f)