import cv2

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read('./model.xml')


def face_detector(image_):
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return image_, []

    for (x, y, w, h) in faces:
        cv2.rectangle(image_, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2)
        region_of_interest = image_[y:y+h, x:x+w]
        region_of_interest = cv2.resize(region_of_interest, dsize=(200, 200))

    return image_, region_of_interest


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = 100*(1 - result[1]/300)
            display_string = '{} Confidence it is user'.format(str(int(confidence)))
        cv2.putText(image, display_string, (100,200), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(250, 120, 255),
                    thickness=2)

        if confidence > 75:
            cv2.putText(image, "Face Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0),
                        thickness=2)
            cv2.imshow('Face Cropper', image)
        else:
            cv2.putText(image, "Face not Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.imshow('Face Cropper', image)

    except Exception as e:
        print(e)
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255),
                    thickness=2)
        cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()




