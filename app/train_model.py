from data_loader import DataLoader
from models import LinearRegression
import config_loader

window_size = config_loader.get_window_size()

# Load ALL of the training data
print("Loading training data...")
X, y = DataLoader.get_windowed_training_data(window_size)
# train the model (and save it to file)
print("Training the model...")
model = LinearRegression()
model.train(X, y, save_model=True)
