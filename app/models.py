# An interface for interacting with machine learning models
class MLModel:
  def fit(daily_cases, daily_vaccinations):
    """Train the model to fit the data"""
    pass

  def predict(days):
    """Predict the given number of days"""
    # We may need to pass in the state data too
    pass 

# TODO: make classes that implement this interface