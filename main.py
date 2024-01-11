
import cv2,math
import mediapipe as mp
import numpy as np
def calculate_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = math.fabs(math.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(RGB)
    landmarks = results.pose_landmarks

    if landmarks is not None:
        # Get the landmarks for the desired points
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate the angles
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Draw the angles on the frame
        cv2.putText(frame, str(round(left_arm_angle, 2)), 
                    tuple(np.multiply([left_elbow.x, left_elbow.y], [frame.shape[1], frame.shape[0]]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(round(right_arm_angle, 2)), 
                    tuple(np.multiply([right_elbow.x, right_elbow.y], [frame.shape[1], frame.shape[0]]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # show the final output
    cv2.imshow('Output', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


