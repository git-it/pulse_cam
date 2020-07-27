pulse-estimator
-----------------------

# how to use:
* must have python 3.7+ installed.
* git clone (or download and unzip this repo)
```
pip install -r requirements.txt
python get_pulse.py
```
That's it.


# What is this project:
1) using OpenCV (cv2) and the well documented _haarcascade_ model idetnify faces from attached webcam
2) estimate forehead location
3) extract optical absorption from forehead*
4) graph and estimate pulse


*Using optical absorption characteristics of (oxy-) haemoglobin ( https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-16-26-21434&id=175396)  we estimate pulse.


this project is based off a multitude of sources, too many to list. So thanks internet ;) you are alwasy there for me.
