# ID-Dec

# Problem Statement: Extracting the information from a document using given input video/image


# Solution
A python project that outputs the extracted information of an input query frame. 
___

* First, we need all the requirements to be installed.
```python
pip install -r requirements.txt
pip install easyocr

```
___ 

* Here is the quick tutorial for running the program 
```python
python detect.py --weights trained_models/obj_det.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
``` 
  ___

# Future Scope
* The Object Detection model was trained on artificial dataset downloaded through Bing and Google APIs. Adding more real time perspective to the training dataset should increase the performance of the model.
* The classification model consist of 2 classes. Given an input query, it must return an output between the 2 class. A 3rd class can be introduced which will be different from the 2 class to generalise the model. 
* The documents detected can be further processed to upgrade the quality of the image that goes into the OCR engine.
* This end to end pipeline which is purely pythonic, can be deployed in flask.
* We can replace the OCR model with the more accurate one. 
