import cv2, os, json, random
import numpy as np
from ageGenderDetect import *

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

## CONFIG
page = 1
zipcode = "11201"

face_counter = 0
detected_gender_list, detected_age_list = [], []
data_path = os.path.abspath(os.getcwd()) + "/data/"

zipcode_map = json.load(open(data_path + "zipmap.json"))
voterCountyDict = {'Bronx': 'Moderately Likely', 'Kings': 'Moderately Likely', 'New York': 'Unikely', 'Queens': 'Highly Likely', 'Richmond': 'Highly Likely'}

def get311Data(zipcode):

    data = json.load(open(data_path + "311.json"))
    try:
        complaintDisplay = data[zipcode]
    except:
        complaintDisplay = "Noise - Residential"
    return complaintDisplay

def getVoterData(zipcode):

    if zipcode in zipcode_map:
        county = zipcode_map[zipcode]['County Name']
        voterStatus = voterCountyDict[county]
    else:
        voterStatus = "Unlikely"

    return voterStatus

def getIncomeData(zipcode):

    income_data = json.load(open(data_path + "income.json"))

    if zipcode in income_data:
        income = income_data[zipcode]
    elif zipcode in zipcode_map:
        county = zipcode_map[zipcode]['County Name']
        income = income_data[county]
    else:
        income = income_data['New York']

    incomeRound = int(income/1000)*1000
    incomeDisplay = "$" + str(incomeRound-10000) + " - " + str(incomeRound+10000)

    return incomeDisplay


def readJson(filePath):
    with open(filePath) as json_file:
        f = json.load(json_file)
    return f

def govImStyling(gray,x,y,w,h):
    im_blur = cv2.GaussianBlur(gray,(7,7),0)
    final_output = cv2.Canny(im_blur, 50, 100)
    return final_output

def govTextStyling(final_output, page, age, gender, zipcode, incomeDisplay, complaintDisplay, voterStatus):

    cv2.rectangle(final_output, (x, y), (x+w, y+h), (255, 255, 255), 4)
    cv2.rectangle(final_output, (0, 0), (frame_width,50), (0,0,0), -1)
    cv2.rectangle(final_output, (0, frame_height-50), (frame_width,frame_height), (0,0,0), -1)
    cv2.putText(final_output, "Gender Prediction: " + str(gender),(50, 700), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)
    cv2.putText(final_output, "Age Estimation: " + str(age),(900, 700),cv2.FONT_HERSHEY_TRIPLEX, 0.75,(255, 255, 255),2, cv2.LINE_AA)
    cv2.putText(final_output, "Location: " + zipcode,(x, int(y - h*0.25)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)

    if page:
        cv2.putText(final_output, "I'm Taxed Therefore I Am",(int(frame_width/4), 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
        cv2.putText(final_output, "Expected Income Range: " + incomeDisplay,(int(x-w*1.1), int(y + h*1.2)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)
        cv2.putText(final_output, "Most Likely 311 Complaint: " + complaintDisplay,(int(x-w*1.1), int(y+h*1.4)),cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 255, 255),4, cv2.LINE_AA)

    else:
        cv2.putText(final_output, "I Vote Therefore I Am",(int(frame_width/4), 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
        cv2.putText(final_output, "Voter Registration Status: " + voterStatus,(int(x-w*1.1), int(y + h*1.2)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)

    return final_output

complaintDisplay = get311Data(zipcode)
incomeDisplay = getIncomeData(zipcode)
voterStatus = getVoterData(zipcode)

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if video_capture.isOpened():
    frame_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float


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
        final_output = govImStyling(gray,x,y,w,h)

        # style final output - text
        final_output = govTextStyling(final_output, page, age, gender, zipcode, incomeDisplay, complaintDisplay, voterStatus)

        # Display the final frame
        cv2.imshow('FaceDetection', final_output)

        # Hotkey - ESC Pressed - End Video Capture
        if k== 27:
            break

        # Hotkey - SPACE pressed - Change Data and Color
        elif k== 32:
            page = not page

        #s pressed
        elif k== 115:
            img_name = "facedetect_webcam_gov.png"
            cv2.imwrite(img_name, final_output)
            print("{} written!".format(img_name))

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
