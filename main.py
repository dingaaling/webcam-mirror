from imutils.video import VideoStream
from flask import Response, Flask, render_template
import threading, argparse, datetime, imutils, cv2, time, os
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
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detected_gender_list, detected_age_list = [], []
facebookDisplay = False
zipcode = "11201"

if facebookDisplay:
    color = facebookStyling.colorSample()
    fbData = facebookStyling.getFacebookData("test")
    # fbData = facebookStyling.getFacebookData("facebook/facebook-jending/")
    adInterestDisplay = facebookStyling.adSampleDisplay(fbData)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_face():
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock, color

# loop over frames from the video stream
    while True:

        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        # frame = imutils.resize(frame, width=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if facebookDisplay:
            frame = facebookStyling.mainStyling(frame, color)
        else:
            edge = govStyling.mainStyling(frame, zipcode)

        # detect face and plot rectangle
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # detect age, gender and peer group for each detected face
        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            if facebookDisplay:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 5)
                frame = facebookStyling.faceStyling(face, frame, x, y, w, h, color)
            else:
                edge = cv2.rectangle(edge, (x, y), (x+w, y+h), (255, 255, 255), 5)

            if len(detected_gender_list)<50:

                gender, age = detectAgeGender(face)
                detected_gender_list.append(gender)
                detected_age_list.append(age)

            genderDisplay, ageDisplay, peerGroupDisplay = ageGenderDisplay(detected_gender_list, detected_age_list)

            if facebookDisplay:
                frame = facebookStyling.peerText(frame, peerGroupDisplay)
                frame = facebookStyling.styleFacebookData(frame, adInterestDisplay, x, y, w, h, color)
            else:
                edge = govStyling.mainTextStyling(edge, ageDisplay, genderDisplay)
                eddge = govStyling.taxStyling(edge, zipcode)

        # acquire the lock, set the output frame, and release the lock
        with lock:
            if facebookDisplay:
                outputFrame = frame.copy()
            else:
                outputFrame = edge.copy()


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
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_face)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
