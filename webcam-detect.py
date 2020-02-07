import cv2, sys, os, json, random
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
padding = 20
face_counter = 0
gender_list, age_list = [], []

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(5 - 10)', '(11 - 15)', '(16 - 24)', '(25 - 32)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
peerGroupList = ['New to the World', 'New to the World', 'Starting Adult Life', 'Starting Adult Life', 'Starting Adult Life', 'Established Adult Life', 'Established Adult Life', 'Established Adult Life']

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

fb_dict = dict()
fb_dict["ads_and_businesses"] = ["ads_interests.json"]

fb_data = dict()
fb_data["ads_interests.json"] = 'topics'

# amazon, facebook, twitter, instagram, slack, spotify, youtube colors
color_list = [(242, 161, 29), (0, 153, 255), (242, 119, 24), (152, 89, 59),(163, 42, 195), (215, 95, 76), (189, 50, 114), (28, 8, 189), (240, 175, 0), (75, 21, 74), (96, 215, 30), (0, 0, 255)]
color = random.sample(color_list, 1)[0]

data_path = os.path.abspath(os.getcwd()) + "/data/"


def readJson(filePath):
    with open(filePath) as json_file:
        f = json.load(json_file)
    return f

def getFacebookData():

    peerGroup, adInterests, adInteracts = [], [], []

    for dir_name in os.listdir(data_path):
        if dir_name.startswith("facebook-j"):
            fbPath = data_path + dir_name + "/"

    for folder in os.listdir(fbPath):
        if folder in fb_dict.keys():

            if folder == 'ads_and_businesses':
                for file in fb_dict[folder]:
                    if file == "ads_interests.json":
                        filePath = fbPath + folder + "/" + file
                        f = readJson(filePath)
                        adInterests = f[fb_data[file]]
                        numAdInterests = len(adInterests)
                        adInterestDisplay = random.sample(adInterests, 3)


    return adInterestDisplay, numAdInterests

def get311Data(zipcode):

    data = json.load( open( data_path + "311.json" ) )
    try:
        complaintDisplay = data[zipcode]
    except:
        complaintDisplay = "Noise - Residential"

    return complaintDisplay


showEdge = 1
zipcode = "10460"
adInterestDisplay, numAdInterests  = getFacebookData()
complaintDisplay = get311Data(zipcode)

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

while (video_capture.isOpened()):

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width, _ = frame.shape

    k = cv2.waitKeyEx(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=15,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face_counter < 50:
            face_counter+=1

            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            gender_list.append(gender)

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            peerGroupDisplay = peerGroupList[agePreds[0].argmax()]
            age_list.append(age)

        else:
            face_counter+=1
            gender = max(set(gender_list), key=gender_list.count)
            age = max(set(age_list), key=age_list.count)
            peerGroupDisplay = peerGroupList[ageList.index(age)]


        im_blur = cv2.GaussianBlur(gray,(7,7),0)

        if showEdge:

            final_output_edge = cv2.Canny(im_blur,50, 100)

            grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(grayscale_face, numAdInterests, 0.005, 10)
            corners = np.int0(corners)

            for corner in corners:
                cx,cy = corner.ravel()
                cv2.circle(final_output_edge,(cx+x,cy+y),2,(255, 255, 255),-1)

            cv2.rectangle(final_output_edge, (x, y), (x+w, y+h), (255, 255, 255), 4)
            cv2.rectangle(final_output_edge, (0, 0), (frame_width,50), (0,0,0), -1)
            cv2.rectangle(final_output_edge, (0, frame_height-50), (frame_width,frame_height), (0,0,0), -1)
            cv2.putText(final_output_edge, "I Vote Therefore I Am",(int(frame_width/2)-200, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
            cv2.putText(final_output_edge, "Gender Prediction: " + str(gender),(50, 700), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)
            cv2.putText(final_output_edge, "Age Estimation: " + str(age),(900, 700),cv2.FONT_HERSHEY_TRIPLEX, 0.75,(255, 255, 255),2, cv2.LINE_AA)

            cv2.putText(final_output_edge, "Location: " + zipcode,(x, y-50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)
            # cv2.putText(final_output_edge, "Voter Status: Not Registered in New York",(x-w+35, int(y + h*0.2)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)
            # cv2.putText(final_output_edge, "Tax Bracket: 90th Percentile",(int(x-w*1.2), int(y + h*1.2)), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255),4, cv2.LINE_AA)
            cv2.putText(final_output_edge, "Most Likely 311 Complaint: " + complaintDisplay,(int(x-w*1.2), int(y+h*1.3)),cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 255, 255),4, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('FaceDetection', final_output_edge)

        else:

            grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(grayscale_face, numAdInterests, 0.005, 10)
            corners = np.int0(corners)
            # Pixelation: Get input size
            im_width, im_height, _ = face.shape
            temp = cv2.resize(face, (20, 20), interpolation=cv2.INTER_LINEAR)
            final_face = cv2.resize(temp, (im_height, im_width), interpolation=cv2.INTER_NEAREST)
            final_output = frame
            final_output[y:y+h, x:x+w] = final_face

            for corner in corners:
                cx,cy = corner.ravel()
                cv2.circle(final_output,(cx+x,cy+y),2,color,-1)

            cv2.rectangle(final_output, (x, y), (x+w, y+h), color, 2)

            cv2.rectangle(final_output, (0, 0), (frame_width,50), color, -1)
            cv2.rectangle(final_output, (0, frame_height-50), (frame_width,frame_height), color, -1)
            cv2.putText(final_output, "I Like Therefore I Am",(int(frame_width/2)-200, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255),4, cv2.LINE_AA)
            cv2.putText(final_output, "Status: " + peerGroupDisplay,(50, 700), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)
            cv2.putText(final_output, "Location: " + "New York City",(800, 700),cv2.FONT_HERSHEY_TRIPLEX, 0.75,(255, 255, 255),2, cv2.LINE_AA)

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
            adInterestDisplay, numAdInterests  = getFacebookData()
            color = random.sample(color_list, 1)[0]

    #s pressed
    elif k== 115:
        img_name = "facedetect_webcam.png"
        if showEdge:
            cv2.imwrite(img_name, final_output_edge)
        else:
            cv2.imwrite(img_name, final_output)

        print("{} written!".format(img_name))


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
