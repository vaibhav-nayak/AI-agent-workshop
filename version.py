import tensorflow as tf
import numpy as np

print("TensorFlow Version:", tf.__version__)
print("NumPy Version:", np.__version__)

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU is available: {gpu_devices}")
else:
    print("GPU not available, TensorFlow will run on CPU.")