import cv2, os, json, random
import numpy as np
from ageGenderDetect import *

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if video_capture.isOpened():
    frame_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

face_counter = 0
detected_gender_list, detected_age_list = [], []

# amazon, facebook, twitter, instagram, slack, spotify, youtube colors
color_list = [(242, 161, 29), (0, 153, 255), (242, 119, 24), (152, 89, 59),(163, 42, 195), (215, 95, 76), (189, 50, 114), (28, 8, 189), (240, 175, 0), (75, 21, 74), (96, 215, 30), (0, 0, 255)]
color = random.sample(color_list, 1)[0]

data_path = os.path.abspath(os.getcwd()) + "/data/"

def readJson(filePath):
    with open(filePath) as json_file:
        f = json.load(json_file)
    return f

def getFacebookData():

    adInterests = []

    for dir_name in os.listdir(data_path):
        if dir_name.startswith("facebook"):
            filePath = data_path + dir_name + "/ads_and_businesses/ads_interests.json"
            if filePath:
                f = readJson(filePath)
                adInterests = f["topics"]
                numAdInterests = len(adInterests)
                adInterestDisplay = random.sample(adInterests, 3)

    return adInterestDisplay, numAdInterests

def facebookImStyling(face, frame, numAdInterests):
    # Pixelate Face
    im_width, im_height, _ = face.shape
    temp = cv2.resize(face, (20, 20), interpolation=cv2.INTER_LINEAR)
    final_face = cv2.resize(temp, (im_height, im_width), interpolation=cv2.INTER_NEAREST)
    final_output = frame.copy()
    final_output[y:y+h, x:x+w] = final_face

    # Detect Corners
    grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(grayscale_face, numAdInterests, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        cx,cy = corner.ravel()
        cv2.circle(final_output,(cx+x,cy+y),2,color,-1)

    return final_output

def facebookTextStyling(final_output, x, y, w, h, peerGroupDisplay, adInterestDisplay):

    cv2.rectangle(final_output, (x, y), (x+w, y+h), color, 2)

    cv2.rectangle(final_output, (0, 0), (frame_width,50), color, -1)
    cv2.rectangle(final_output, (0, frame_height-50), (frame_width,frame_height), color, -1)
    cv2.putText(final_output, "I Like Therefore I Am",(int(frame_width/2)-200, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
    cv2.putText(final_output, "Status: " + peerGroupDisplay,(50, 700), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)
    cv2.putText(final_output, "Current Location: " + "New York City",(800, 700),cv2.FONT_HERSHEY_TRIPLEX, 0.75,(255, 255, 255),2, cv2.LINE_AA)

    cv2.putText(final_output, adInterestDisplay[0],(x, int(y - h*0.25)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)
    cv2.putText(final_output, adInterestDisplay[1],(int(x-w*1.05), int(y + h*0.5)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)
    cv2.putText(final_output, adInterestDisplay[2],(int(x+w), int(y + h*0.75)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)

    return final_output


adInterestDisplay, numAdInterests  = getFacebookData()

while (video_capture.isOpened()):

    # pull webcam frames and convert to grayscale to detect face
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    k = cv2.waitKeyEx(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=15,
        minSize=(60, 60)
    )

    # detect age, gender and peer group for each detected face
    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if len(detected_gender_list)<50:

            gender, age = detectAgeGender(face)
            detected_gender_list.append(gender)
            detected_age_list.append(age)

        genderDisplay, ageDisplay, peerGroupDisplay = ageGenderDisplay(detected_gender_list, detected_age_list)

        # style final output - visual
        final_output = facebookImStyling(face, frame, numAdInterests)

        # style final output - text
        final_output = facebookTextStyling(final_output, x, y, w, h, peerGroupDisplay, adInterestDisplay)

        # Display the final frame
        cv2.imshow('FaceDetection', final_output)

        # Hotkey - ESC Pressed - End Video Capture
        if k== 27:
            break

        # Hotkey - SPACE pressed - Change Data and Color
        elif k== 32:
            adInterestDisplay, numAdInterests  = getFacebookData()
            color = random.sample(color_list, 1)[0]

        #s pressed
        elif k== 115:
            img_name = "facedetect_webcam_fb.png"
            cv2.imwrite(img_name, final_output)
            print("{} written!".format(img_name))

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
