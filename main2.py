# TensorFlow and tf.keras
import tensorflow as tf
import tensorrt
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())

