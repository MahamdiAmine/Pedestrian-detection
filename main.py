import eel
import argparse
import cv2 as cv
import numpy as np
from yolo_utils import infer_image, show_image

eel.init('website')

model_path = './yolov3-coco/'
weights_path = './yolov3-coco/yolov3.weights'
config_path = './yolov3-coco/yolov3.cfg'
vid_output_path = './output.mp4'
labels_path = './yolov3-coco/coco-labels'
confidence = 0.5
threshold = 0.3
show_Time = False


@eel.expose
def update_params(weightsPath, outputPath, confidenceValue, threshold_Value, showTime):
    global weights_path, vid_output_path, confidence, threshold, show_Time
    weights_path, vid_output_path = weightsPath, outputPath
    confidence, threshold, show_Time = confidenceValue, threshold_Value, showTime
    print("Params updated")
    print("out :" + vid_output_path)


@eel.expose
def test(videoPath):
    labels = open(labels_path).read().strip().split('\n')

    # Initializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configuration to form the pre-trained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    height, width = None, None
    try:
        vid = cv.VideoCapture(videoPath)
        writer = None
    except Exception:
        raise Exception("Video cannot be loaded!\n Please check the path provided!")

    finally:
        while True:
            grabbed, frame = vid.read()

            # Checking if the complete video is read
            if not grabbed:
                break

            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, show_Time,
                                            float(confidence), float(threshold))

            if writer is None:
                # Initialize the video writer
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter(vid_output_path, fourcc, 30,
                                        (frame.shape[1], frame.shape[0]), True)

            writer.write(frame)
            cv.imshow('Video', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        writer.release()
        vid.release()


@eel.expose
def webCam():
    print("Starting Inference on WebCam")
    # FLAGS, unparsed = pass_args()
    labels = labels_path
    # Init colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # Load the weights and configuration to form the pre-trained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1]
                   for i in net.getUnconnectedOutLayers()]
    # Infer real-time on webcam
    count = 0
    vid = cv.VideoCapture(0)
    while True:
        _, frame = vid.read()
        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                                                    height, width, frame, colors, labels, confidences,
                                                                    classids, idxs)
            count += 1
        else:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors,
                                                                    labels, show_Time, confidence, threshold, boxes,
                                                                    confidences, classids, idxs,
                                                                    infer=False)
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
