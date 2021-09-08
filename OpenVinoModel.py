import logging as log

import cv2
import numpy as np
from openvino.inference_engine import IECore


class OpenVinoModel:
    def __init__(self, model_path, input_size=(112, 112), device_name="CPU"):
        self.input_size = input_size
        # Load model
        ie = IECore()
        model = model_path
        log.info(f"Loading network:\n\t{model}")
        net = ie.read_network(model=model)

        # Defind in, out
        self.input_blob = next(iter(net.input_info))
        out_blob = next(iter(net.outputs))

        # Get input
        n, c, h, w = net.input_info[self.input_blob].input_data.shape
        print("Input shape", n, c, h, w)

        # Load model to plugin
        print("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name=device_name)

    def predict(self, image):
        # Inference
        image = np.expand_dims(cv2.resize(image, self.input_size), axis=0)
        image = image.transpose((0, 3, 1, 2))
        res = self.exec_net.infer(inputs={self.input_blob: image})
        return list(res.values())
