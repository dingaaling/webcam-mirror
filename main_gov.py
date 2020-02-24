from flask import Response, Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import threading, argparse, imutils, cv2, time, os, json, jsonify
from imutils.video import VideoStream
from ageGenderDetect import *
import govStyling


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the face detection model
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

form_dict = dict()
form_dict['zipcode'] = '11201'
form_dict['voterStatus'] = ''
form_dict['stopEvent'] = False

# main face detection and video data styling function
def detect_face():
    # grab global references to the video stream, output frame, and lock variables
    global outputFrame, lock, frame

    # set global age, gender, gov data variables
    detected_gender_list, detected_age_list = [], []

# loop over frames from the video stream
    while True:

        # check if stop event initiatied to stop video and end thread
        if form_dict['stopEvent']:
            vs.stop()
            print("stop event started")
            return

        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        # frame = imutils.resize(frame, width=1150)
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

        # detect and display age, gender and gov data for each detected face
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

@app.route("/heartbeat")
def heartbeat():
    return jsonify({"status": "healthy"})

# if "More Data" button clicked, sample for more data
@app.route("/sample", methods=['POST'])
def sample_data():

    action = request.form['action']
    form_dict['taxStatus'] = not form_dict['taxStatus']
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

# if "Start Over" button clicked, return home and set stop event
@app.route("/home", methods=['POST'])
def home():

    action = request.form['action']
    form_dict['stopEvent'] = True

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

# if "Save Image" button clicked, save portrait with random hash
@app.route("/save", methods=['POST'])
def saveImage():

    hash = govStyling.randomHash()
    img_name = "portraits/gov-" + hash + ".png"
    cv2.imwrite(img_name, outputFrame)
    print(img_name)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


@app.route("/video_feed")
def video_feed():

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# main video page displaying face detection video stream
# start video stream and face detection thread
@app.route('/video', methods=['GET'])
def video():

    # return the response generated along with the specific media
    # type (mime type)
    global vs, t

    form_dict['stopEvent'] = False

    # initialize the video stream and allow the camera sensor to warmup
    vs = VideoStream(src=0)
    vs.start()

    # start a thread that will perform face detection
    t = threading.Thread(target=detect_face)
    t.daemon = True
    t.start()

    print('(video feed) number of current threads is ', threading.active_count(), threading.enumerate())

    return render_template("video.html", ip=args["ip"], port=args["port"])

# home page - display form
@app.route('/', methods=['GET'])
def index():

    form_dict['stopEvent'] = True
    print('(index) number of current threads is ', threading.active_count(), threading.enumerate())

    return render_template("index_gov.html", ip=args["ip"], port=args["port"])

# home page - process form data
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
    try:
        form_dict['voterStatus'] = govStyling.getVoterStatus(form_dict)
    except:
        form_dict['voterStatus'] = "Not Registered in NYC"

    print(form_dict)
    print('(process form) number of current threads is ', threading.active_count(), threading.enumerate())
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
    print('(index) number of current threads is ', threading.active_count(), threading.enumerate())

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
