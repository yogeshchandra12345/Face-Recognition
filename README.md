#### Face Authentication 
Face Authentication using LBPH (Local Binary Pattern Histogram Classifier).

Steps to build:
1. Prepare dataset using command $python prepare_data.py
2. Train the model using openCV inbuilt LBPH FaceRecognizer. $python train_model.py
3. Run the classifier to authenticate the person captured by webcam. $python classify.py

Demo:

###### When the face is present in database.
<img src="https://raw.githubusercontent.com/yogeshchandra12345/Face-Recognition/master/face_recognize_correct.png" width="700" height="350">


###### When the face is not present in database.
<img src="https://raw.githubusercontent.com/yogeshchandra12345/Face-Recognition/master/face_not_matched.png" width="700" height="350">
