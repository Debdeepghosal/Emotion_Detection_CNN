from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')

#Haar cascade path
face_cascade_path = 'haarcascade_frontalface_default .xml'

# Loading the Haar cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the cascade classifier was loaded successfully
if faceCascade.empty():
    print("Error: Could not load the Haar cascade classifier.")
    exit()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    res, frame = cap.read()
    if not res:
        break

    height, width, channels = frame.shape
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray_image ,scaleFactor=1.3, minNeighbors=5)


    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_image[y - 5:y + h + 5, x - 5:x + w + 5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            cv2.imshow('Detected Face', cv2.resize(roi_gray, (w * 2, h * 2)))
            image_pixels = np.array(roi_gray, dtype=np.float32)
            image_pixels /= 255.00
            image_pixels = np.expand_dims(image_pixels, axis=0)
            # Assuming 'model' is defined and loaded for emotion detection
            predictions = model.predict(image_pixels)
            max_pred_index = np.argmax(predictions[0])
            emotion = ("angry","disgusted","fearful","happy","neutral","sad","surprised")
            emotion_predicted = emotion[max_pred_index]
            cv2.putText(frame, f'Emotion: {emotion_predicted}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error: {e}")
        continue

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
