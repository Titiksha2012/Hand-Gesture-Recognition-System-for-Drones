import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.layers import TFSMLayer
from sklearn.metrics import classification_report

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load gesture recognizer model
model = TFSMLayer('mp_hand_gesture', call_endpoint='serving_default')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

print("Classes:", classNames)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Lists to store predictions and ground truths
y_true = []
y_pred = []

print("Press keys 0–9 to record the ground truth label after each prediction. Press 'q' to quit.")

while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand landmark detection
    result = hands.process(framergb)
    className = ''
    classID = -1

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y] for lm in handslms.landmark]
            landmarks = np.array(landmarks).flatten().astype(np.float32)
            input_tensor = tf.convert_to_tensor([landmarks])

            prediction_dict = model(input_tensor)
            prediction = list(prediction_dict.values())[0].numpy()
            classID = np.argmax(prediction)
            className = classNames[classID]

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    # Show predicted label
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF

    # Check if user pressed 0–9 (ground truth input)
    if 48 <= key <= 57:  # ASCII values for 0–9
        true_label = key - 48
        if classID != -1:
            y_true.append(true_label)
            y_pred.append(classID)
            print(f"GT: {classNames[true_label]}, Predicted: {className}")

    elif key == ord('q'):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()

# Show evaluation metrics
if y_true and y_pred:
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=classNames))
else:
    print("No ground truth data was recorded.")
