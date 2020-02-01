import cv2, sys, os, json, random, time
from datetime import datetime
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
img_counter = 0
padding = 20

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

fb_dict = dict()
fb_dict["about_you"] = "friend_peer_group.json"
fb_dict["ads_and_businesses"] = ["ads_interests.json", "advertisers_you've_interacted_with.json"]
fb_dict["location"] = "location_history.json"

fb_data = dict()
fb_data["friend_peer_group.json"] = 'friend_peer_group'
fb_data["ads_interests.json"] = 'topics'
fb_data["location_history.json"] = 'location_history'
fb_data["advertisers_you've_interacted_with.json"] = 'history'


data_path = os.path.abspath(os.getcwd()) + "/data/"


def readJson(filePath):
    with open(filePath) as json_file:
        f = json.load(json_file)
    return f

def getFacebookData():

    peerGroup, adInterests, adInteracts, location = [], [], [], []

    for dir_name in os.listdir(data_path):
        if dir_name.startswith("facebook"):
            fbPath = data_path + dir_name + "/"

    for folder in os.listdir(fbPath):
        if folder in fb_dict.keys():

            if folder =="about_you":
                filePath = fbPath + folder + "/" + fb_dict[folder]
                f = readJson(filePath)
                peerGroup = f[fb_data[fb_dict[folder]]]

            elif folder == 'location':
                filePath = fbPath + folder + "/" + fb_dict[folder]
                f = readJson(filePath)
                location = f[fb_data[fb_dict[folder]]]

            elif folder == 'ads_and_businesses':
                for file in fb_dict[folder]:
                    if file == "ads_interests.json":
                        filePath = fbPath + folder + "/" + file
                        f = readJson(filePath)
                        adInterests = f[fb_data[file]]
                    elif file == "advertisers_you've_interacted_with.json":
                        filePath = fbPath + folder + "/" + file
                        f = readJson(filePath)
                        adInteracts = f[fb_data[file]]

    peerGroupDisplay = peerGroup
    locationDisplay = random.sample(location, 1)[0]
    adInterestDisplay = random.sample(adInterests, 1)[0]
    adInteractDisplay = random.sample(adInteracts, 1)[0]

    return peerGroupDisplay, locationDisplay, adInterestDisplay, adInteractDisplay


showEdge = 0
peerGroupDisplay, locationDisplay, adInterestDisplay, adInteractDisplay  = getFacebookData()
ts = int(locationDisplay['creation_timestamp'])
color=np.random.rand(3,)*255

while (video_capture.isOpened()):

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    k = cv2.waitKeyEx(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)

        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        # print("Gender : {}".format(gender))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        # print("Age Output : {}".format(agePreds))
        # print("Age : {}".format(age))


        im_blur = cv2.GaussianBlur(frame,(7,7),0)

        grayscale = cv2.cvtColor(im_blur, cv2.COLOR_BGR2GRAY)
        # edge = cv2.Canny(grayscale, 75, 125)
        edge = cv2.Laplacian(grayscale,cv2.CV_64F)

        grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(grayscale_face, 100, 0.01, 10)
        corners = np.int0(corners)


        if showEdge:

            cv2.rectangle(edge, (x, y), (x+w, y+h), (255, 64, 64), 2)

            cv2.putText(edge, "Location: 11201",(x+w+30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,64,64),2, cv2.LINE_AA)
            cv2.putText(edge, "Most Likely 311 Complaint: Noise - Residential",(x+w+30, int(y + h*0.25)),cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,64,64),2, cv2.LINE_AA)
            cv2.putText(edge, "Voter Status: Not Registered in New York",(x+w+30, int(y + h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,64,64),2, cv2.LINE_AA)
            cv2.putText(edge, "Tax Bracket: ?",(x+w+30, int(y + h*0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,64,64),2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('FaceDetection', edge)

        else:

            for corner in corners:
                cx,cy = corner.ravel()
                cv2.circle(im_blur,(cx+x,cy+y),3,color,-1)

            cv2.rectangle(im_blur, (x, y), (x+w, y+h), color, 2)
            cv2.putText(im_blur, "Gender Prediction: " + str(gender),(int(x-w*1.2), int(y + h*0.25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2, cv2.LINE_AA)
            cv2.putText(im_blur, "Age Estimation: " + str(age),(int(x-w*1.2), int(y + h*0.5)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,color,2, cv2.LINE_AA)

            cv2.putText(im_blur, "Friend Peer Group: " + peerGroupDisplay,(x+w+30, int(y + h*0.25)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,color,2, cv2.LINE_AA)
            cv2.putText(im_blur, "Ad Interest: " + adInterestDisplay,(x+w+30, int(y + h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2, cv2.LINE_AA)
            cv2.putText(im_blur, adInteractDisplay['action'] + ": " + adInteractDisplay['title'],(x+w+30, int(y + h*0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2, cv2.LINE_AA)
            cv2.putText(im_blur, "Detected location: " + locationDisplay['name'] + ' on ' + datetime.utcfromtimestamp(ts).strftime('%m/%d/%Y'), (x+w+30, int(y + h*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2, cv2.LINE_AA)


            # Display the resulting frame
            cv2.imshow('FaceDetection', im_blur)

    #ESC Pressed
    if k== 27:
        break

    #c pressed
    elif k==99:
        showEdge = not showEdge

    #SPACE pressed
    elif k== 32:

        if showEdge==0:
            peerGroupDisplay, locationDisplay, adInterestDisplay, adInteractDisplay  = getFacebookData()
            ts = int(locationDisplay['creation_timestamp'])
            color=np.random.rand(3,)*255
            print(color)

    #s pressed
    elif k== 115:
        img_name = "facedetect_webcam_{}.png".format(img_counter)
        cv2.imwrite(img_name, edge)
        print("{} written!".format(img_name))
        img_counter += 1


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
