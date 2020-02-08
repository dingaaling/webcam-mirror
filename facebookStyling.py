import cv2, os, random, json
import numpy as np

# amazon, facebook, twitter, instagram, slack, spotify, youtube colors
color_list = [(242, 161, 29), (0, 153, 255), (242, 119, 24), (152, 89, 59),(163, 42, 195), (215, 95, 76), (189, 50, 114), (28, 8, 189), (240, 175, 0), (75, 21, 74), (96, 215, 30), (0, 0, 255)]
dataPath = os.path.abspath(os.getcwd()) + "/data/"

def colorSample():
    color = random.sample(color_list, 1)[0]
    return color

def adSampleDisplay(adList):
    adInterestDisplay = random.sample(adList, 3)
    return adInterestDisplay

def mainStyling(frame, color):
    frame_height, frame_width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (frame_width,50), color, -1)
    cv2.rectangle(frame, (0, frame_height-50), (frame_width,frame_height), color, -1)
    cv2.putText(frame, "I Like Therefore I Am",(int(frame_width/4), 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
    cv2.putText(frame, "Location: " + "New York City",(int(0.67*frame_width), int(frame_height-20)),cv2.FONT_HERSHEY_TRIPLEX, 0.75,(255, 255, 255),2, cv2.LINE_AA)

    return frame

def faceStyling(face, frame, x, y, w, h, color):

    # Pixelate Face
    im_width, im_height, _ = face.shape
    temp = cv2.resize(face, (20, 20), interpolation=cv2.INTER_LINEAR)
    final_face = cv2.resize(temp, (im_height, im_width), interpolation=cv2.INTER_NEAREST)
    final_output = frame.copy()
    final_output[y:y+h, x:x+w] = final_face

    # Detect Corners
    grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(grayscale_face, 200, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        cx,cy = corner.ravel()
        cv2.circle(final_output,(cx+x,cy+y),3,color,-1)

    return final_output

def peerText(frame, peerGroupDisplay):

    frame_height, frame_width, _ = frame.shape
    cv2.putText(frame, "Status: " + peerGroupDisplay,(10, int(frame_height-20)), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)

    return frame

def readJson(filePath):
    with open(filePath) as json_file:
        f = json.load(json_file)
    return f

def getFacebookData(fbDir):

    adInterests = []
    fbPath = dataPath + fbDir
    if os.path.exists(fbPath):
        f = readJson(fbPath + "/ads_and_businesses/ads_interests.json")
        adInterests = f["topics"]

    # ADD PROPUBLICA DATA
    else:
        adInterests = ["Test", "Test", "Test", "Test", "Test"]

    return adInterests

def styleFacebookData(frame, adInterestDisplay, x, y, w, h, color):

    short = min((word for word in adInterestDisplay if word), key=len)
    long = max((word for word in adInterestDisplay if word), key=len)
    med = list(set(adInterestDisplay) - set([short, long]))[0]

    cv2.putText(frame, long,(x, int(y - h*0.25)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)
    cv2.putText(frame, med,(int(x-w*1.05), int(y + h*0.5)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)
    cv2.putText(frame, short,(int(x+w), int(y + h*0.75)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)

    return frame
