import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf

threshold = 0.8
offset=20 
imgSize = 224
class_names = ["1", "2", "3", "4", "5"]
display_names = {
    "1": "Select",
    "2": "Cancel",
    "3": "Execute/Confirm",
    "4": "Next",
    "5": "Previous"
}
#load model
model = tf.keras.models.load_model("hand_model.h5")
model.make_predict_function()


cap = cv.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

while True:
    succes, img = cap.read()
    if not succes:
        break
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img)
    #convert back to BGR to display
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255   #we multyiply by 255 to make pixels white rather than black

            h, w, _ = img.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = max(0, int(min(x_coords) * w)-offset)
            x_max = min(w, int(max(x_coords) * w)+offset)
            y_min = max(0, int(min(y_coords) * h)-offset)
            y_max = min(h, int(max(y_coords) * h)+offset)


            if x_max > x_min and y_max > y_min:
                imgCrop = img[y_min:y_max, x_min:x_max]
                if imgCrop.size == 0: continue
                #cv.imshow("ImageCrop", imgCrop)

                h_crop, w_crop, _ = imgCrop.shape

                if h_crop > w_crop:
                    scale = imgSize / h_crop
                    w_new = min(imgSize, max(1, math.ceil(w_crop * scale)))
                    imgResize = cv.resize(imgCrop, (w_new, imgSize))

                    wGap = (imgSize - w_new) // 2
                    imgWhite[:, wGap:wGap + w_new] = imgResize

                else:
                    scale = imgSize / w_crop
                    h_new = min(imgSize, max(1, math.ceil(h_crop * scale)))
                    imgResize = cv.resize(imgCrop, (imgSize, h_new))

                    hGap = (imgSize - h_new) // 2
                    imgWhite[hGap:hGap + h_new, :] = imgResize


                #prediction
                img_input = imgWhite / 255.0
                img_input = np.expand_dims(img_input, axis=0)
                pred = model.predict(img_input)[0]
                index = np.argmax(pred)
                confidence = pred[index]

                if confidence < threshold:
                    display_gesture_name = "Unknown"
                else:
                    gesture = class_names[index]
                    display_gesture_name = display_names[gesture]
                cv.putText(img, f'Prediction: {display_gesture_name}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("Live Prediction", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
