# webcam-mirror
==============================

* Use a webcam as a mirror to view your data identities
* Flask app that accepts basic user information to display your Facebook and New York City data over a live OpenCV facial detection visualization
* An exploration tool that allows you to explore your own data, as well as aggregated data of other New York residents and crowdsourced FB ads dataset from Propublica (https://www.propublica.org/datastore/dataset/facebook-ad-categories)

# Demos

## Facebook Mirror interface demo - uploading my own Facebook Data
![](https://github.com/dingaaling/webcam-mirror/blob/master/static/FB-Jen.gif)

## Facebook Mirror interface demo - using Propublica FB Ads Dataset (**Not my data!**)
![](https://github.com/dingaaling/webcam-mirror/blob/master/static/FB-Propublica.gif)

## New York City Mirror interface demo 
![](https://github.com/dingaaling/webcam-mirror/blob/master/static/NYC.gif)


Project Organization
------------

    ├── README.md
    ├── main_fb.py              < - flask app to run fb web interface  >
    ├── main_gov.py             < - flask app to run gov web interface  >
    ├── ageGenderDetect.py      < - helper functions to detect face/age/gender  >
    ├── facebookStyling.py      < - helper functions for styling facebook video stream  >
    ├── govStyling.py           < - helper functions for styling gov video stream  >
    ├── models                  <- ML prototxt and caffe models for face/age/gender detection >
    ├── static                  <- CSS files >
    │   ├── styles                  
    │        ├── main.css                
    ├── templates               <- HTML files >
    │   ├── index_fb.html       < - display gov form UI >
    │   ├── index_gov.html      < - display facebook form UI >
    │   ├── video.html          < - display video stream UI >
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
