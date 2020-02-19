import cv2, os, random, json, mechanize
import numpy as np
from bs4 import BeautifulSoup


dataPath = os.path.abspath(os.getcwd()) + "/data/"
zipcode_map = json.load(open(dataPath + "zipmap.json"))
countyBoroughDict = {'Bronx': 'Bronx', 'Kings': 'Brooklyn', 'New York': 'Manhattan', 'Queens': 'Queens', 'Richmond': 'Staten Island'}
partyDict = {'DEM': 'Democrat', 'REP': 'Republican', 'CON': 'Conservative', 'WOR': 'Working Families', 'GRE': 'Green',
            'LBT': 'Libertarian', 'IND': 'Independence', 'SAM': 'Serve America Movement', 'OTH': 'Other', 'BLK': 'No party affiliation'}

def randomHash():
    hash = random.getrandbits(128)
    return str(hash)

def mainStyling(frame, zipcode, taxStatus):
    frame_height, frame_width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (frame_width,50), (0,0,0), -1)
    cv2.rectangle(frame, (0, frame_height-50), (frame_width,frame_height), (0,0,0), -1)
    if taxStatus:
        cv2.putText(frame, "I'm Taxed Therefore I Am", (int(frame_width/4), 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
    else:
        cv2.putText(frame, "I Vote Therefore I Am", (int(frame_width/4), 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
    cv2.putText(frame, "Location: " + zipcode,(int(frame_width/2.5), 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)

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
    corners = cv2.goodFeaturesToTrack(grayscale_face, 150, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        cx,cy = corner.ravel()
        cv2.circle(final_output,(cx+x,cy+y),3,color,-1)

    return final_output

def mainTextStyling(frame, age, gender):
    frame_height, frame_width, _ = frame.shape
    cv2.putText(frame, "Gender Prediction: " + str(gender),(10, int(frame_height-20)), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)
    cv2.putText(frame, "Age Estimation: " + str(age),(int(0.67*frame_width), int(frame_height-20)),cv2.FONT_HERSHEY_TRIPLEX, 0.75,(255, 255, 255),2, cv2.LINE_AA)
    return frame

def taxStyling(frame, zipcode):
    frame_height, frame_width, _ = frame.shape
    complaintDisplay = get311Data(zipcode)
    incomeDisplay = getIncomeData(zipcode)
    cv2.putText(frame, "Expected Income Range: " + incomeDisplay,(int(frame_width/6), int(frame_height*0.75)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)
    cv2.putText(frame, "Most Likely 311 Complaint: " + complaintDisplay,(int(frame_width/6), int(frame_height*0.85)),cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 255, 255),4, cv2.LINE_AA)

    return frame

def voteStyling(frame, voterStatus, zipcode):
    frame_height, frame_width, _ = frame.shape
    complaintDisplay = get311Data(zipcode)
    incomeDisplay = getIncomeData(zipcode)
    cv2.putText(frame, "Voter Status: " + voterStatus,(int(frame_width/6), int(frame_height*0.85)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)


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


def getBorough(zipcode):

    if zipcode in zipcode_map:
        county = zipcode_map[zipcode]['County Name']
        borough = countyBoroughDict[county]
    else:
        borough = None

    return borough

def getVoterStatus(form_dict):


    br = mechanize.Browser()
    br.set_handle_robots(False)
    br.addheaders = [('User-agent', 'Firefox')]
    br.set_debug_http(True)
    br.open('https://www.nycvotersearch.com/')
    response = br.response()
    br.form = list(br.forms())[0]

    br["LastName"] = form_dict['lastName']
    br["FirstName"] = form_dict['firstName']
    br["ZipCode"] = form_dict['zipcode']
    br["bMonth"] = form_dict['birthMonth']
    br["bDay"] = form_dict['birthDay']
    br["bYear"] = form_dict['birthYear']
    br["County"] = [form_dict['borough']]

    response = br.submit()
    response_html = response.read()
    soup = BeautifulSoup(response_html, 'lxml')
    response_str = soup.text

    if "Your search did not match any record on file" in response_str:
        voterStatus = "Not Registered in NYC"
    else:
        response_str = response_str.replace("\n", " ")
        response_str = response_str.replace("\r", " ")
        response_str = response_str.replace("                    ", " ")
        response_str = response_str.replace("     ", " ")

        party_ind_start, stat_ind_start, stat_ind_end = response_str.find("Party:"), response_str.find("Status:"), response_str.find("VOTER DISTRICT INFORMATION")
        party_acro = str(response_str[party_ind_start+6:stat_ind_start])
        party_acro = str.strip(party_acro)
        activeStatus = response_str[stat_ind_start+8:stat_ind_end]

        if party_acro in partyDict:
            party = partyDict[party_acro]

        voterStatus = str(activeStatus) + "   " + str(party)

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
