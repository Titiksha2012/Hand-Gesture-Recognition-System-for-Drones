# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.layers import TFSMLayer

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model using TFSMLayer
model = TFSMLayer('mp_hand_gesture', call_endpoint='serving_default')  # Confirm endpoint if needed

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    className = ''

    # Post-process the result
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handslms.landmark:
                landmarks.append([lm.x, lm.y])  # Use normalized coordinates

            # Prepare input for the model
            landmarks = np.array(landmarks).flatten().astype(np.float32)
            input_tensor = tf.convert_to_tensor([landmarks])  # Shape: (1, 42)

            # Predict gesture
            prediction_dict = model(input_tensor)
            prediction = list(prediction_dict.values())[0].numpy()
            print("Prediction values:", prediction)
            classID = np.argmax(prediction)
            print("Predicted class index:", classID)
            className = classNames[classID]


            # Draw landmarks
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    # Show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Output", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()



