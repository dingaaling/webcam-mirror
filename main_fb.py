from flask import Response, Flask, render_template, request, flash, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import threading, argparse, imutils, cv2, time, os
from zipfile import ZipFile
from imutils.video import VideoStream
from ageGenderDetect import *
import facebookStyling


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

UPLOAD_FOLDER = os.path.abspath(os.getcwd()) + '/data/facebook/'
ALLOWED_EXTENSIONS = {'zip', 'none'}

# initialize a flask object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

detected_gender_list, detected_age_list = [], []
frameCount = 0

fb_dict = dict()

def saveImage(frame):

    hash = facebookStyling.randomHash()
    img_name = "portraits/facebook-" + hash + ".png"
    print(img_name)
    cv2.imwrite(img_name, frame)


def newFacebookDisplay(fbfilePath):

    color = facebookStyling.colorSample()
    fbAds = facebookStyling.getFacebookData(fbfilePath)
    adInterestDisplay = facebookStyling.adSampleDisplay(fbAds)

    return color, adInterestDisplay


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

        frame = facebookStyling.mainStyling(frame, fb_dict['color'])
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

            cv2.rectangle(frame, (x, y), (x+w, y+h), fb_dict['color'], 5)
            frame = facebookStyling.faceStyling(face, frame, x, y, w, h, fb_dict['color'])

            if len(detected_gender_list)<50:

                gender, age = detectAgeGender(face)
                detected_gender_list.append(gender)
                detected_age_list.append(age)

            genderDisplay, ageDisplay, peerGroupDisplay = ageGenderDisplay(detected_gender_list, detected_age_list)
            frame = facebookStyling.peerText(frame, peerGroupDisplay)
            frame = facebookStyling.styleFacebookData(frame, fb_dict['ads'], x, y, w, h, fb_dict['color'])

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
    color, adInterestDisplay = newFacebookDisplay(fb_dict['fbFolder'])
    fb_dict['color'] = color
    fb_dict['ads'] = adInterestDisplay

    return action

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/video/<filename>')
def video(filename):

    if filename !="none":
        fb_dict['fbFolder'] = UPLOAD_FOLDER + filename.split(".")[0]

        with ZipFile(UPLOAD_FOLDER + filename, 'r') as zipObj:
            zipObj.extractall(fb_dict['fbFolder'])
        os.remove(UPLOAD_FOLDER + filename)
    else:
        fb_dict['fbFolder'] = "none"

    return render_template("video.html")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
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

    return render_template("index_fb.html")

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

    # start a thread that will perform face detection
    t = threading.Thread(target=detect_face)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)


# release the video stream pointer
vs.stop()
