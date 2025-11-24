import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import time

offset=20 
imgSize = 224
folder = "Data/1"
counter=0

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

                cv.imshow("WhiteImage", imgWhite)




    cv.imshow("Image", img)
    key= cv.waitKey(1)
    if key == ord("s"): #if key "s" is pressed then take an image and save it into folder path
        if np.std(imgWhite) < 5:
            print("No hand detected")
            continue
        else:
            counter += 1
            print(counter)
            cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
    elif key == ord("1"):
        folder = "Data/1"
    elif key == ord("2"):
        folder = "Data/2"
    elif key == ord("3"):
        folder = "Data/3"
    elif key == ord("4"):
        folder = "Data/4"
    elif key == ord("5"):
        folder = "Data/5"
    elif key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
