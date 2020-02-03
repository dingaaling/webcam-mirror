import cv2, sys, os, json, random, time
from datetime import datetime
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
img_counter = 0
padding = 20

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(4-6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']

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
    adInterestDisplay = random.sample(adInterests, 3)
    adInteractDisplay = random.sample(adInteracts, 1)[0]

    return peerGroupDisplay, locationDisplay, adInterestDisplay, adInteractDisplay


showEdge = 1
peerGroupDisplay, locationDisplay, adInterestDisplay, adInteractDisplay  = getFacebookData()
ts = int(locationDisplay['creation_timestamp'])
color=np.random.rand(3,)*255

face_counter = 0
gender_list, age_list = [], []

while (video_capture.isOpened()):

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    k = cv2.waitKeyEx(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=9,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face_counter < 50:
            face_counter+=1
            ageNet = cv2.dnn.readNet(ageModel, ageProto)
            genderNet = cv2.dnn.readNet(genderModel, genderProto)

            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            # if genderPreds[0][0] < 0.7:
            #     genderPreds[0][0] = 0
            gender = genderList[genderPreds[0].argmax()]
            gender_list.append(gender)

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            age_list.append(age)

        else:
            gender = max(set(gender_list))
            age = max(set(age_list))


        im_blur = cv2.GaussianBlur(frame,(7,7),0)
        grayscale = cv2.cvtColor(im_blur, cv2.COLOR_BGR2GRAY)

        if showEdge:

            edge = cv2.Canny(grayscale,50, 100)
            final_output_edge = cv2.dilate(edge,None)


            cv2.rectangle(final_output_edge, (x, y), (x+w, y+h), (255, 255, 255), 4)
            cv2.putText(final_output_edge, "Location: 11201",(x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),5, cv2.LINE_AA)
            cv2.putText(final_output_edge, "Voter Status: Not Registered in New York",(x-w+35, int(y + h*0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255),5, cv2.LINE_AA)
            cv2.putText(final_output_edge, "Tax Bracket: 75 Percentile",(int(x-w*0.5), int(y + h*1.15)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255),5, cv2.LINE_AA)
            cv2.putText(final_output_edge, "Most Likely 311 Complaint: Noise - Residential",(x-w, int(y+h*1.4)),cv2.FONT_HERSHEY_SIMPLEX, 1.25,(255, 255, 255),5, cv2.LINE_AA)


            # Display the resulting frame
            cv2.imshow('FaceDetection', final_output_edge)

        else:

            grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(grayscale_face, 70, 0.005, 10)
            corners = np.int0(corners)
            # Pixelation: Get input size
            im_width, im_height, _ = face.shape
            temp = cv2.resize(face, (20, 20), interpolation=cv2.INTER_LINEAR)
            final_face = cv2.resize(temp, (im_height, im_width), interpolation=cv2.INTER_NEAREST)
            final_output = frame
            final_output[y:y+h, x:x+w] = final_face

            for corner in corners:
                cx,cy = corner.ravel()
                cv2.circle(final_output,(cx+x,cy+y),3,color,-1)

            cv2.rectangle(final_output, (x, y), (x+w, y+h), color, 2)
            cv2.putText(final_output, "Gender Prediction: " + str(gender),(10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,2, cv2.LINE_AA)
            cv2.putText(final_output, "Age Estimation: " + str(age),(450, 700),cv2.FONT_HERSHEY_SIMPLEX, 0.6,color,2, cv2.LINE_AA)
            cv2.putText(final_output, "Friend Peer Group: " + peerGroupDisplay,(900, 700),cv2.FONT_HERSHEY_SIMPLEX, 0.6,color,2, cv2.LINE_AA)

            cv2.putText(final_output, adInterestDisplay[0],(x, int(y - h*0.25)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)
            cv2.putText(final_output, adInterestDisplay[1],(int(x-w*1.05), int(y + h*0.5)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)
            cv2.putText(final_output, adInterestDisplay[2],(int(x+w), int(y + h*0.75)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color,4, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('FaceDetection', final_output)

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

    #s pressed
    elif k== 115:
        img_name = "facedetect_webcam_{}.png".format(img_counter)
        if showEdge:
            cv2.imwrite(img_name, final_output_edge)
        else:
            cv2.imwrite(img_name, final_output)

        print("{} written!".format(img_name))
        img_counter += 1


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
