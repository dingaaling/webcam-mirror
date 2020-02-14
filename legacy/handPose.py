import cv2
import numpy as np


protoFile = "models/pose_deploy.prototxt"
weightsFile = "models/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
threshold = 0.2

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def detectHand(frame):

    numPoints = 0
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    aspect_ratio = frameWidth/frameHeight

    # inHeight = 368
    inHeight = 250
    inWidth = int(((aspect_ratio*inHeight)*8)//8)

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    numPoints = sum(x is not None for x in points)

    return numPoints
