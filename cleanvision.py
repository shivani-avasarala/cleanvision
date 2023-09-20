#!/usr/bin/env python
# coding: utf-8

# # MONODEPTH on OpenVINO IR Model
# 
# This notebook demonstrates Monocular Depth Estimation with MidasNet in OpenVINO. Model information: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/midasnet/midasnet.md

# <img src="monodepth.gif">

# ### What is Monodepth?
# Monocular Depth Estimation is the task of estimating scene depth using a single image. It has many potential applications in robotics, 3D reconstruction, medical imaging and autonomous systems. For this demo, we use a neural network model called [MiDaS](https://github.com/intel-isl/MiDaS) which was developed by the Intelligent Systems Lab at Intel. Check out their research paper to learn more. 
# 
# R. Ranftl, K. Lasinger, D. Hafner, K. Schindler and V. Koltun, ["Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer,"](https://ieeexplore.ieee.org/document/9178977) in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2020.3019967.

# ## Preparation 

# ### Imports

import os
import time
import urllib
from pathlib import Path

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np


from IPython.display import (
    HTML,
    FileLink,
    Pretty,
    ProgressBar,
    Video,
    clear_output,
    display,
)
from openvino.inference_engine import IECore

DEVICE = "CPU"
td_MODEL_FILE = "models/horizontal-text-detection-0001.xml"
tr_MODEL_FILE = "models/text-recognition-0012.xml"

td_model_name = os.path.basename(td_MODEL_FILE)
td_model_xml_path = Path(td_MODEL_FILE).with_suffix(".xml")

tr_model_name = os.path.basename(tr_MODEL_FILE)
tr_model_xml_path = Path(tr_MODEL_FILE).with_suffix(".xml")

def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())


def load_image(path: str):
    """
    Loads an image from `path` and returns it as BGR numpy array. `path`
    should point to an image file, either a local filename or an url.
    """
    if path.startswith("http"):
        # Set User-Agent to Mozilla because some websites block
        # requests with User-Agent Python
        request = urllib.request.Request(
            path, headers={"User-Agent": "Mozilla/5.0"}
        )
        response = urllib.request.urlopen(request)
        array = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(array, -1)  # Loads the image as BGR
    else:
        image = cv2.imread(path)
    return image


def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

# --------------------------- SETTING UP IE FOR BOTH MODELS
# td_ie = IECore()
# td_net = td_ie.read_network(
#     str(td_model_xml_path), str(td_model_xml_path.with_suffix(".bin"))
# )
# td_exec_net = td_ie.load_network(network=td_net, device_name=DEVICE)

# td_input_key = list(td_exec_net.input_info)[0]
# td_output_key = list(td_exec_net.outputs.keys())[0]

tr_ie = IECore()
tr_net = tr_ie.read_network(
    str(tr_model_xml_path), str(tr_model_xml_path.with_suffix(".bin"))
)
tr_exec_net = tr_ie.load_network(network=tr_net, device_name=DEVICE)

tr_input_key = list(tr_exec_net.input_info)[0]
tr_output_key = list(tr_exec_net.outputs.keys())[0]
# ----------------------------------------------------------


# td_network_input_shape = td_exec_net.input_info[td_input_key].tensor_desc.dims
# td_network_image_height, td_network_image_width = td_network_input_shape[2:]

tr_network_input_shape = tr_exec_net.input_info[tr_input_key].tensor_desc.dims
tr_network_image_height, tr_network_image_width = tr_network_input_shape[2:]

get_ipython().run_line_magic('matplotlib', 'inline')

# WEBCAM CAPTURE --------------------------------------

cap = cv2.VideoCapture(0)
while(True):
    # Capture the video frame
    # by frame
    ret, image = cap.read()
    if not ret:
        raise ValueError(f"Image frame could not be read.")
    # Display the resulting frame
    cv2.imshow('frame', image)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("input_image.jpg", image)
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

# -----------------------------------------------------

IMAGE_FILE = "input_image.jpg"
image = load_image(IMAGE_FILE)
input_frame_height, input_frame_width = image.shape[:2]

# resize to input shape for network
resized_image = cv2.resize(image, (tr_network_image_width, tr_network_image_height))
gray_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

# # reshape image to network input shape NCHW
# input_image = np.expand_dims(np.transpose(gray_resized_image, (2, 0, 1)), 0)
input_image = np.expand_dims(gray_resized_image, 0)
input_image = np.expand_dims(input_image, 0)

tr_result = tr_exec_net.infer(inputs={tr_input_key: input_image})[tr_output_key]
print(" result # of dimensions: ", tr_result.ndim)
print(" result shape of dimensions: ", tr_result.shape)

characters = "0123456789abcdefghijklmnopqrstuvwxyz#"
text = []

for i in tr_result:
    conf = i[0]
    last = ''
    index = np.argmax(conf)
    if characters[index] != '#':
        if last != characters[index]:
            text.append(characters[index])
            last = characters[index]

print(text)


# print("network dims: ", " h: ", network_image_height, " w: ", network_image_width)
# print("shape: ", network_input_shape)

# xmin = int(output[3]*network_image_width)
# ymin = int(output[4]*network_image_height)
# xmax = int(output[5]*network_image_width)
# ymax = int(output[6]*network_image_height)




# print(xmin, " ", ymin, " ", xmax, " ", ymax)

# cv2.rectangle(resized_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
# output_image = cv2.resize(resized_image, (input_video_frame_width, input_video_frame_height))
# cv2.imwrite("output_image.jpg", output_image)


# convert network result of disparity map to an image that shows
# distance as colors
# result_image = convert_result_to_image(result)
# print(result_image)
# # resize back to original image shape. cv2.resize expects shape
# # in (width, height), [::-1] reverses the (height, width) shape to match this.
# # change : result_image = cv2.resize(result_image, image.shape[:2][::-1])

# # fig, ax = plt.subplots(1, 2, figsize=(20, 15))
# # ax[0].imshow(to_rgb(image))
# # ax[1].imshow(result_image)

# # Filename
# newimagename = 'savedImage.jpg'
  
# # Using cv2.imwrite() method
# # Saving the image
# cv2.imwrite(newimagename, result_image)





