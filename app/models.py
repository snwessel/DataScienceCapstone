import pandas as pd
from fbprophet import Prophet

# An interface for interacting with machine learning models
class MLModel:
  def fit(self, daily_cases, daily_vaccinations):
    """Train the model to fit the data"""
    pass

  def predict(self, num_days):
    """Predict the given number of days"""
    # We may need to pass in the state data too
    pass 


# An implementation of the interface using Facebook's prophet model.
# Note: this model currently only looks at the case counts (not vaccinations).
# I'm not sure if there is a way to factor the vaccinations into this particular model.
class ProphetModel(MLModel):

  def __init__(self):
    self.model = Prophet()

  def fit(self, daily_cases, daily_vaccinations):
    """Train the model to fit the data"""
    self.model.fit(daily_cases)

  def predict(self, num_days):
    """Predict the given number of days"""
    # the prediction dataframe contains the following fields: DS, YHAT, YHAT, YHAT_UPPER
    # the dataframe should also include the prior values
    future = self.model.make_future_dataframe(periods=365)
    forecast = self.model.predict(future)

    # temporary: plot the output:
    self.model.plot_components(forecast)
    return future