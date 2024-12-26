import os

# Configuration parameters
DATA_DIR = "imagenet-dataset"   # Default data directory
NUM_EPOCHS = 110                 # Number of training epochs
BATCH_SIZE = 5120               # Batch size for training
LEARNING_RATE = 0.05            # Learning rate for the optimizer

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory: {DATA_DIR}")
