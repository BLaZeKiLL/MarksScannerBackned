import tensorflow as tf
import json
import cv2
import os

data = {
  "tf_version": tf.__version__,
  "cwd": os.path.join(os.getcwd(), 'python', 'my_model_f.h5')
}

print(json.dumps(data))
