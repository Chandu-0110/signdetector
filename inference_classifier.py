import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary

labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'F',4:'I',5:'L',6:'O',7:'R',8:'V',9:'Y'}

# Function to process hand gestures
def process_gestures(frame):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    sign_detected = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == 42:  # 21 (x, y) coordinates for a single hand
            prediction = model.predict([np.asarray(data_aux)])
            sign_detected = labels_dict[int(prediction[0])]

    return frame, sign_detected


def main():
    st.title("SIGN-TO-TEXT ALPHABET CONVERSION SYSTEM")
    st.markdown("## Interact Now 🤖!")
    image_placeholder = st.empty()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Streamlit loop
    while True:
        ret, frame = cap.read()
        if ret:
            frame_processed, sign = process_gestures(frame)
            if sign:
                cv2.putText(frame_processed, f"Sign Detected: {sign}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2, cv2.LINE_AA)
                st.write(sign)
            else:
                cv2.putText(frame_processed, "No sign detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)

            image_placeholder.image(frame_processed, channels="BGR", use_column_width=True)


if __name__ == "__main__":
    main()
