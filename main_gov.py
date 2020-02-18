from flask import Response, Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import threading, argparse, imutils, cv2, time, os, json
from zipfile import ZipFile
from imutils.video import VideoStream
from ageGenderDetect import *
import govStyling


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()


# initialize a flask object
app = Flask(__name__)

faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

detected_gender_list, detected_age_list = [], []
frameCount = 0

form_dict = dict()
form_dict['zipcode'] = '11201'
form_dict['voterStatus'] = ''

def saveImage(frame):

    hash = govStyling.randomHash()
    img_name = "portraits/gov-" + hash + ".png"
    print(img_name)
    cv2.imwrite(img_name, frame)


def detect_face():
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock, frameCount

# loop over frames from the video stream
    while True:

        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=1150)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

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

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,0), 5)
            frame = govStyling.faceStyling(face, frame, x, y, w, h, (0,0,0))

            if len(detected_gender_list)<50:

                gender, age = detectAgeGender(face)
                detected_gender_list.append(gender)
                detected_age_list.append(age)

            genderDisplay, ageDisplay, peerGroupDisplay = ageGenderDisplay(detected_gender_list, detected_age_list)
            frame = govStyling.mainTextStyling(frame, ageDisplay, genderDisplay)
            if form_dict['taxStatus']:
                frame = govStyling.taxStyling(frame, form_dict['zipcode'])
            else:
                frame = govStyling.voteStyling(frame, form_dict['voterStatus'], form_dict['zipcode'])

        frameCount+=1
        if frameCount > 1000:
            saveImage(frame)
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

@app.route("/sample", methods=['POST'])
def sample_data():

    action = request.form['action']
    form_dict['taxStatus'] = not form_dict['taxStatus']


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)

    # start a thread that will perform face detection
    t = threading.Thread(target=detect_face)
    t.daemon = True
    t.start()

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/video', methods=['GET'])
def video():

    return render_template("video.html")

@app.route('/', methods=['GET'])
def index():
    return render_template("index_gov.html")

@app.route('/', methods=['POST'])
def process_form():
    form_dict['zipcode'] = request.form['zipCode']
    form_dict['firstName'] = str(request.form['firstName'])
    form_dict['lastName'] = str(request.form['lastName'])
    form_dict['birthDay'] = str(request.form['birthDay'])
    form_dict['birthMonth'] = str(request.form['birthMonth'])
    form_dict['birthYear'] = str(request.form['birthYear'])
    borough = govStyling.getBorough(form_dict['zipcode'])
    form_dict['borough'] = str(borough)
    form_dict['voterStatus'] = govStyling.getVoterStatus(form_dict)
    print(form_dict)
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

if __name__ == "__main__":
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")


    args = vars(ap.parse_args())
    form_dict['taxStatus'] = False

    # initialize the video stream and allow the camera sensor to warmup
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)


# release the video stream pointer
vs.stop()
