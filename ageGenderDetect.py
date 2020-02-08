import cv2
import numpy as np


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
peerGroupList = ['New to the World', 'Starting Adult Life', 'Established Adult Life']


ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


def detectAgeGender(face):

    blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return gender, age


def ageGenderDisplay(detected_gender_list, detected_age_list):

    genderDisplay = max(set(detected_gender_list), key=detected_gender_list.count)
    ageDisplay = max(set(detected_age_list), key=detected_age_list.count)

    if int(ageDisplay[-3:-1]) < 10:
        peerGroupDisplay = peerGroupList[0]
    elif int(ageDisplay[-3:-1]) < 40:
        peerGroupDisplay = peerGroupList[1]
    else:
        peerGroupDisplay = peerGroupList[2]

    return genderDisplay, ageDisplay, peerGroupDisplay
