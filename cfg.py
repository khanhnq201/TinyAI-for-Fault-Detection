import torch
#--- Data loading ---
BASE_PATH = 'CWRU-dataset-main'
SAMPLE_LENGTH = 1024
PREPROCESSING = True
OVERLAPPING_RATIO = 0.25
IMBALANCED_MODE = False
IMBALANCED_RATIO = 20
RANDOM_STATE = 42

#--- Model parameter --- 
CNN1D_INPUT = False
MODEL_TYPE = 'student' #studen/ teacher/ 1D
# NUM_CLASSES = 4
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 200
LEARNING_RATE = 0.0001
BATCH_SIZE = 128

# CLASS_NAMES = ['Normal', 'B', 'IR', 'OR']
CLASS_NAMES = ['Normal', 'IR', 'OR'] 
