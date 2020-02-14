from imutils.video import VideoStream
from flask import Response, Flask, render_template, request
import threading, argparse, imutils, cv2, time, os
from ageGenderDetect import *
import facebookStyling
import govStyling

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)


# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

detected_gender_list, detected_age_list = [], []
frameCount = 0

form_dict = dict()
form_dict['zipcode'] = '11201'
form_dict['voterStatus'] = ''

fb_dict = dict()


def saveImage(frame, facebookDisplay):

    hash = facebookStyling.randomHash()
    if facebookDisplay:
        img_name = "portraits/facebook-" + hash + ".png"
    else:
        img_name = "portraits/gov-" + hash + ".png"
    print(img_name)
    cv2.imwrite(img_name, frame)


def newFacebookDisplay():

    color = facebookStyling.colorSample()
    # fbAds, fbAdvertisers = facebookStyling.getFacebookData("none")
    fbAds, fbAdvertisers = facebookStyling.getFacebookData("facebook/facebook-jending/")
    adInterestDisplay, advertiserDisplay = facebookStyling.adSampleDisplay(fbAds, fbAdvertisers)

    return color, adInterestDisplay, advertiserDisplay


@app.route("/")
def index():

    return render_template("index.html")

def detect_face():
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock, facebookDisplay, frameCount

# loop over frames from the video stream
    while True:

        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=1150)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if facebookDisplay:
            frame = facebookStyling.mainStyling(frame, fb_dict['color'])
        else:
            frame = govStyling.mainStyling(frame, form_dict['zipcode'], form_dict['taxStatus'])

        # detect face and plot rectangle
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # detect age, gender and peer group for each detected face
        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            if not facebookDisplay:
                fb_dict['color'] = (0,0,0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), fb_dict['color'], 5)
            frame = facebookStyling.faceStyling(face, frame, x, y, w, h, fb_dict['color'])

            if len(detected_gender_list)<50:

                gender, age = detectAgeGender(face)
                detected_gender_list.append(gender)
                detected_age_list.append(age)

            genderDisplay, ageDisplay, peerGroupDisplay = ageGenderDisplay(detected_gender_list, detected_age_list)

            if facebookDisplay:
                frame = facebookStyling.peerText(frame, peerGroupDisplay)
                frame = facebookStyling.styleFacebookData(frame, fb_dict['ads'], x, y, w, h, fb_dict['color'])
            else:
                frame = govStyling.mainTextStyling(frame, ageDisplay, genderDisplay)
                if form_dict['taxStatus']:
                    frame = govStyling.taxStyling(frame, form_dict['zipcode'])
                else:
                    frame = govStyling.voteStyling(frame, form_dict['voterStatus'], form_dict['zipcode'])

        frameCount+=1
        if frameCount > 1000:
            saveImage(frame, facebookDisplay)
            break

        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/form", methods=['POST'])
def process_form():

    user_zipcode = request.form['zipCode']

    form_dict['zipcode'] = user_zipcode
    form_dict['firstName'] = str(request.form['firstName'])
    form_dict['lastName'] = str(request.form['lastName'])
    form_dict['birthDay'] = str(request.form['birthDay'])
    form_dict['birthMonth'] = str(request.form['birthMonth'])
    form_dict['birthYear'] = str(request.form['birthYear'])
    borough = govStyling.getBorough(user_zipcode)
    form_dict['borough'] = str(borough)

    form_dict['voterStatus'] = govStyling.getVoterStatus(form_dict)

    return user_zipcode


@app.route("/sample", methods=['POST'])
def sample_data():

    action = request.form['action']
    if facebookDisplay:
        color, adInterestDisplay, advertiserDisplay = newFacebookDisplay()
        fb_dict['color'] = color
        fb_dict['ads'] = adInterestDisplay
    else:
        form_dict['taxStatus'] = not form_dict['taxStatus']

    return action


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")

    args = vars(ap.parse_args())

    if args["port"] == 8080:
        facebookDisplay = True
        color, adInterestDisplay, advertiserDisplay = newFacebookDisplay()
        fb_dict['color'] = color
        fb_dict['ads'] = adInterestDisplay

    else:
        facebookDisplay = False
        form_dict['taxStatus'] = False

    # start a thread that will perform face detection
    t = threading.Thread(target=detect_face)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)


# release the video stream pointer
vs.stop()
