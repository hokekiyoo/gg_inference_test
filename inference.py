import logging
import os

from dlr import DLRModel
from PIL import Image
import numpy as np

import greengrasssdk
import camera
import utils
import json
import base64
import requests
from io import BytesIO

# Create MQTT client
mqtt_client = greengrasssdk.client('iot-data')
# Initialize logger
customer_logger = logging.getLogger(__name__)

# Initialize example Resnet-50 model
model_resource_path = os.environ.get('MODEL_PATH', '/pytorch-compiled')
input_shape = {'0': [1, 3, 32, 32]}
output_shape = [1, 10]
dlr_model = DLRModel(model_resource_path, input_shape, output_shape, 'cpu')
iot_client = greengrasssdk.client('iot-data')
labels= ["airplane", "automobile", "bird", "cat", "deer",
         "dog", "frog", "horse", "ship", "truck"]

def predict(image_data):
    r"""
    Predict image with DLR. The result will be published
    to MQTT topic '/cifar10/predictions'.
    :param image: numpy array of the Image inference with.
    """

    # calc prediction scores
    flattened_data = image_data.astype(np.float32).flatten()
    prediction_scores = dlr_model.run({'0' : flattened_data}).squeeze()
    
    # sort(high-scored order)
    pair = [(score, labels[index]) for index, score in enumerate(prediction_scores)]
    pair = sorted(pair, reverse = True)
    
    # for debug
    p_str = ""
    for s,i in pair:
        print(i,s)
        p_str += "label: {} \t\tscore{}\n".format(i,s)
    send_mqtt_message(p_str)

    # max score and labels
    max_class = pair[0][1]
    label = labels[max_class]
    max_score = pair[0][0]
    result_str = "class:{} score:{}".format(label,max_score)
    send_mqtt_message(
        'Prediction Result: {}'.format(result_str))
    return label, max_score

def send_mqtt_message(message):
    r"""
    Publish message to the MQTT topic:
    '/resnet-50/predictions'.

    :param message: message to publish
    """
    mqtt_client.publish(topic='/cifar10/predictions',
                        payload=message)


# Upload Image Data and Prediction Result
def send_prediction_results(imgdata,label, score):
    send_topic = '/GG_img2S3/result'
    data = {
           "data": imgdata,
           "label": label,
           "score": score
        }
    messageToPublish = data
    print(messageToPublish)
    iot_client.publish(topic=send_topic, payload=json.dumps(messageToPublish))


def predict_from_cam():
    r"""
    Predict with the photo taken from your pi camera.
    """
    send_mqtt_message("Taking a photo...")
    my_camera = camera.Camera()
    imagebinary = my_camera.capture_image()
    image = Image.open(imagebinary)
    image_data = utils.transform_image(image)

    send_mqtt_message("Start predicting...")
    max_class, max_score = predict(image_data)

    # DataEncode
    image64 = base64.b64encode(imagebinary.getvalue())
    image_str = image64.decode("utf-8")

    send_prediction_results(image_str, max_class, float(max_score))

    return imagebinary

# The lambda to be invoked in Greengrass
def handler(event, context):
    try:
        predict_from_cam()
    except Exception as e:
        customer_logger.exception(e)
        send_mqtt_message('Exception occurred during prediction. Please check logs for troubleshooting: /greengrass/ggc/var/log.')

