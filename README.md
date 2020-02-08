# webcam-mirror
==============================

Use a webcam as a mirror for your data identity.

* Run main.py to start the app and input data
* After data submission, view your Facebook, Government, and other demographics data displayed over a live OpenCV facial detection display

Project Organization
------------

    ├── README.md
    ├── main.py (in dev)      < - flask app to display web interface to input zip code and fb data and start displays >
    ├── gov-display.py        < - display gov data >
    ├── facebook-display.py   < - display facebook data >
    ├── data (in dev)         <- datasets used for app >
    │   ├── fb_data           <- Folder that contains ProPublica FB Ads data or uploaded personal fb ads data >
    │   ├──   ├── propublica  <- Propublica  B Ads dataset>
    │   ├── 311.json          <- Dataset of 311 complaint data, by zipcode >
    │   ├── income.json       <- Dataset of average income data, by zipcode, county >
    │   ├── zipmap.json       <- Dataset that maps zipcode to county >
--------

## Getting this running on your computer

Requirements:
- Python3, OS
- OpenCV
- Numpy
- Random
- JSON

To get started (while the flask app is in development), run:

```
python facebook-display.py
```
or
```
python gov-display.py
```

Controls:
- Space bar: change data displayed
- Escape key: end the display
- s key: save image
