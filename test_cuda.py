import numpy as np
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))

print(tf.test.is_gpu_available())