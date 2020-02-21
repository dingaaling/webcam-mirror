# webcam-mirror
==============================

Use a webcam as a mirror for your data identity.

* Run python `main_gov.py --ip 0.0.0.0 --port 8080` or `python main_fb.py --ip 0.0.0.0 --port 8080` to start the app and input data
* After data submission, view your Facebook, Government, and other demographics data displayed over a live OpenCV facial detection display

Project Organization
------------

    ├── README.md
    ├── main_fb.py              < - flask app to run fb web interface  >
    ├── main_gov.py             < - flask app to run gov web interface  >
    ├── gov-display.py          < - display gov data >
    ├── facebook-display.py     < - display facebook data >
    ├── models                  <- ML prototxt and caffe models for face/age/gender detection >
    ├── static                  <- CSS files >
        ├── styles                  
            ├── main.css                
    ├── templates               <- HTML files >
        ├── index_fb.html 
        ├── index_gov.html 
        ├── video.html 
    ├── portraits               <- Placeholder for data portraits >
    ├── data                    <- datasets used for app >
    │   ├── fb_data             <- Folder that contains ProPublica FB Ads data or uploaded personal fb ads data >
    │   ├──   ├── facebook      <- Placeholder for personal dataset>
    │   ├── 311.json            <- Dataset of 311 complaint data, by zipcode >
    │   ├── income.json         <- Dataset of average income data, by zipcode, county >
    │   ├── zipmap.json         <- Dataset that maps zipcode to county >
    │   ├── fb_propublica.json  <- Propublica FB ads dataset >

--------

## Getting this running on your computer

Requirements:
```
pip install requirements.txt
```

To get started (while the flask app is in development), run:

```
python main_fb.py --ip 0.0.0.0 --port 8080
```
or
```
python main_gov.py --ip 0.0.0.0 --port 8080
```
