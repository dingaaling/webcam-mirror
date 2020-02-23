from flask import Response, Flask, render_template, request, flash, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import threading, argparse, imutils, cv2, time, os, json
from zipfile import ZipFile
from imutils.video import VideoStream
from ageGenderDetect import *
import facebookStyling


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# set data path for FB uploads and accepted extension type
UPLOAD_FOLDER = os.path.abspath(os.getcwd()) + '/data/facebook/'
ALLOWED_EXTENSIONS = {'zip', 'none'}

# initialize a flask object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# initialize the face detection model
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# set global age, gender, fb data variables
detected_gender_list, detected_age_list = [], []
frameCount = 0

fb_dict = dict()
fb_dict['stopEvent'] = False


# sample for new color and FB data
def newFacebookDisplay(fbfilePath):

    color = facebookStyling.colorSample()
    fbAds = facebookStyling.getFacebookData(fbfilePath)
    adInterestDisplay = facebookStyling.adSampleDisplay(fbAds)

    return color, adInterestDisplay


# Check for Allowed file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# main face detection and video data styling function
def detect_face():
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock, frameCount, frame

# loop over frames from the video stream
    while True:

        # check if stop event initiatied to stop video and end thread
        if fb_dict['stopEvent']:
            vs.stop()
            print("stop event started")
            return

        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        # frame = imutils.resize(frame, width=1150)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        frame = facebookStyling.mainStyling(frame, fb_dict['color'])
        # detect face and plot rectangle
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # detect and display age, gender and fb data for each detected face
        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            cv2.rectangle(frame, (x, y), (x+w, y+h), fb_dict['color'], 5)
            frame = facebookStyling.faceStyling(face, frame, x, y, w, h, fb_dict['color'])

            if len(detected_gender_list)<50:

                gender, age = detectAgeGender(face)
                detected_gender_list.append(gender)
                detected_age_list.append(age)

            genderDisplay, ageDisplay, peerGroupDisplay = ageGenderDisplay(detected_gender_list, detected_age_list)
            frame = facebookStyling.peerText(frame, peerGroupDisplay)
            frame = facebookStyling.styleFacebookData(frame, fb_dict['ads'], x, y, w, h, fb_dict['color'])


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

# if "More Data" button clicked, sample for more data
@app.route("/sample", methods=['POST'])
def sample_data():

    action = request.form['action']
    color, adInterestDisplay = newFacebookDisplay(fb_dict['fbFolder'])
    fb_dict['color'] = color
    fb_dict['ads'] = adInterestDisplay

    return action

# if "Start Over" button clicked, return home and set stop event
@app.route("/home", methods=['POST'])
def home():

    action = request.form['action']
    fb_dict['stopEvent'] = True

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

# if "Save Image" button clicked, save portrait with random hash
@app.route("/save", methods=['POST'])
def saveImage():

    hash = facebookStyling.randomHash()
    img_name = "portraits/facebook-" + hash + ".png"
    cv2.imwrite(img_name, outputFrame)
    print(img_name)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


@app.route("/video_feed")
def video_feed():

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# main video page displaying face detection video stream
# start video stream and face detection thread
@app.route('/video/<filename>')
def video(filename):

    # return the response generated along with the specific media
    # type (mime type)
    global vs, t

    fb_dict['stopEvent'] = False

    # initialize the video stream and allow the camera sensor to warmup
    vs = VideoStream(src=0)
    vs.start()

    if filename !="none":
        fb_dict['fbFolder'] = UPLOAD_FOLDER + filename.split(".")[0]

        with ZipFile(UPLOAD_FOLDER + filename, 'r') as zipObj:
            zipObj.extractall(fb_dict['fbFolder'])
        os.remove(UPLOAD_FOLDER + filename)
    else:
        fb_dict['fbFolder'] = "none"

    # start a thread that will perform face detection
    t = threading.Thread(target=detect_face)
    t.daemon = True
    t.start()

    print('(video feed) number of current threads is ', threading.active_count(), threading.enumerate())


    return render_template("video.html", ip=args["ip"], port=args["port"])

# home page - display form
@app.route('/', methods=['GET'])
def index():

    fb_dict['stopEvent'] = True
    print('(index) number of current threads is ', threading.active_count(), threading.enumerate())

    return render_template("index_fb.html")

# home page - process upload form
@app.route('/', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        filename = "none"
        return redirect(url_for('video', filename=filename))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('video', filename=filename))

    print('(index) number of current threads is ', threading.active_count(), threading.enumerate())
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

if __name__ == "__main__":
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")

    args = vars(ap.parse_args())

    color, adInterestDisplay = newFacebookDisplay('none')
    fb_dict['color'] = color
    fb_dict['ads'] = adInterestDisplay


    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)


# release the video stream pointer
vs.stop()
