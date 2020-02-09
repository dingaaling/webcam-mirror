import cv2, os, random, json
import numpy as np

dataPath = os.path.abspath(os.getcwd()) + "/data/"
zipcode_map = json.load(open(dataPath + "zipmap.json"))
voterCountyDict = {'Bronx': 'Moderately Likely', 'Kings': 'Moderately Likely', 'New York': 'Unikely', 'Queens': 'Highly Likely', 'Richmond': 'Highly Likely'}

def mainStyling(frame, zipcode):
    frame_height, frame_width, _ = frame.shape
    # blur = cv2.GaussianBlur(frame,(7,7),0)
    # edge = cv2.Canny(blur, 50, 100)
    cv2.rectangle(frame, (0, 0), (frame_width,50), (0,0,0), -1)
    cv2.rectangle(frame, (0, frame_height-50), (frame_width,frame_height), (0,0,0), -1)
    cv2.putText(frame, "I'm Taxed Therefore I Am", (int(frame_width/4), 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
    cv2.putText(frame, "Location: " + zipcode,(int(frame_width/2.5), 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)

    return frame

def mainTextStyling(frame, age, gender):
    frame_height, frame_width, _ = frame.shape
    cv2.putText(frame, "Gender Prediction: " + str(gender),(10, int(frame_height-20)), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)
    cv2.putText(frame, "Age Estimation: " + str(age),(int(0.67*frame_width), int(frame_height-20)),cv2.FONT_HERSHEY_TRIPLEX, 0.75,(255, 255, 255),2, cv2.LINE_AA)
    return frame

def taxStyling(frame, zipcode):
    frame_height, frame_width, _ = frame.shape
    complaintDisplay = get311Data(zipcode)
    incomeDisplay = getIncomeData(zipcode)
    cv2.putText(frame, "Expected Income Range: " + incomeDisplay,(int(frame_width/5), int(frame_height*0.75)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)
    cv2.putText(frame, "Most Likely 311 Complaint: " + complaintDisplay,(int(frame_width/5), int(frame_height*0.85)),cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 255, 255),4, cv2.LINE_AA)

    return frame

def readJson(filePath):
    with open(filePath) as json_file:
        f = json.load(json_file)
    return f

def get311Data(zipcode):

    data = json.load(open(dataPath + "311.json"))
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

    income_data = json.load(open(dataPath + "income.json"))

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
