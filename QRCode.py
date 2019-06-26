import io
import pyqrcode
from base64 import b64encode
import eel
import argparse
import cv2 as cv
import numpy as np
from yolo_utils import infer_image, show_image
eel.init('website')


@eel.expose
def dummy(dummy_param):
    print("I got a parameter: ", dummy_param)
    return "string_value", 1, 1.2, True, [1, 2, 3, 4], {"name": "eel"}
@eel.expose
def echo():
    print("I got a parameter: ")
@eel.expose
def generate_qr():
    print("1111111111")
    FLAGS, unparsed = pass_args()
    labels = open(FLAGS.labels).read().strip().split('\n')
    # Initializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configuration to form the pre-trained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1]
                   for i in net.getUnconnectedOutLayers()]

    # If both image and video files are given then raise error
    if FLAGS.image_path is None and FLAGS.video_path is None:
        print('Neither path to an image or path to video provided')
        print('Starting Inference on Webcam')
    # Infer real-time on webcam
    count = 0
    print("Hereeeeeeeeeee")
    vid = cv.VideoCapture(0)
    while True:
        _, frame = vid.read()
        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                                                    height, width, frame, colors, labels, FLAGS)
            count += 1
        else:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                                                    height, width, frame, colors, labels, FLAGS,
                                                                    boxes, confidences, classids, idxs, infer=False)
            count = (count + 1) % 6

        cv.imshow('webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()


def pass_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path', type=str, default='./yolov3-coco/',
                        help='The directory where the model weights and configuration'
                             ' files are.')

    parser.add_argument('-w', '--weights', type=str, default='./yolov3-coco/yolov3.weights',
                        help='Path to the file which contains the weights for YOLOv3.')

    parser.add_argument('-cfg', '--config', type=str, default='./yolov3-coco/yolov3.cfg',
                        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-i', '--image-path', type=str,
                        help='The path to the image file')

    parser.add_argument('-v', '--video-path', type=str,
                        help='The path to the video file')

    parser.add_argument('-vo', '--video-output-path', type=str, default='./output.avi',
                        help='The path of the output video file')

    parser.add_argument('-l', '--labels', type=str,
                        default='./yolov3-coco/coco-labels',
                        help='Path to the file having the labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                        help='The model will reject boundaries which has a' +
                             'probability less than the confidence value.' +
                             'default: 0.5')

    parser.add_argument('-th', '--threshold', type=float, default=0.3,
                        help='The threshold to use when applying the Non-Max Suppresion')

    parser.add_argument('--download-model', type=bool, default=False,
                        help='Set to True, if the model weights and configurations ' +
                             'are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')

    return parser.parse_known_args()


eel.start('index.html')
