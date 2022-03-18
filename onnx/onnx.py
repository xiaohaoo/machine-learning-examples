import time

import cv2
import numpy as np
import onnxruntime

start_time = time.time()

session = onnxruntime.InferenceSession('onnx.onnx')

if __name__ == '__main__':
    img = cv2.imread('../img.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 127.5 - 1
    print(np.expand_dims(img, 0).shape)
    print(session.get_outputs()[0].name)
    start_time = time.time()
    outputs = session.run(None, {'input_1': np.expand_dims(img, 0)})
    print(np.argmax(outputs))
    print(time.time()-start_time )
