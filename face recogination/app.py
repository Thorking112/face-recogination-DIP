import cv2
import mediapipe as mp
import pyautogui
import numpy as np
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
font = cv2.FONT_HERSHEY_SIMPLEX
def display_message(frame, text, position, color=(0, 255, 0), size=1):
    cv2.putText(frame, text, position, font, size, color, 2, cv2.LINE_AA)
def take_screenshot():
    pyautogui.screenshot('screenshot.png')
    print("Screenshot taken!")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
                display_message(frame, "Taking Screenshot!", (20, 30), color=(0, 0, 255), size=1)
                take_screenshot()
            
            if landmarks.landmark[mp_hands.HandLandmark.WRIST].y < index_tip.y and \
                    landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < index_tip.y:
                display_message(frame, "Open Hand Detected!", (20, 90), color=(0, 255, 255), size=1)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
