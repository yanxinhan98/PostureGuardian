import cv2
import mediapipe as mp
import math
import numpy as np
import time
from plyer import notification

RDV = 52
RAT = 48
ANG1 = -15
ANG2 = 20

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def calculate_relative_distance(distance, shoulder_width):
    # Calculate the relative distance as a percentage
    return (distance / shoulder_width) * 100

def calculate_shoulder_angle_y(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx))

def calculate_face_angle(shoulder_landmark, face_landmark):
    # Calculate the vector between shoulder and face landmarks
    vector = (face_landmark.x - shoulder_landmark.x, face_landmark.y - shoulder_landmark.y, face_landmark.z - shoulder_landmark.z)

    # Calculate the angle between the vector and the horizontal plane
    angle = math.degrees(math.atan2(vector[2], math.sqrt(vector[0] ** 2 + vector[1] ** 2)))

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Initialize the FaceMesh and Pose models
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
shoulder_width = None  # Variable to store shoulder width
# Define the frame skipping interval
frame_skip_interval = 150  # Example: capture every 5th frame

frame_count = 0

pre_status = None
status = None 
while True:
    # Read the current frame
    frame_count += 1
    # time.sleep(3)
    # Skip frames if necessary
    # print(frame_count, frame_count % frame_skip_interval)
    success, image = cap.read()

    if not success:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with the FaceMesh model
    results_face_mesh = face_mesh.process(image_rgb)

    # Process the image with the Pose model
    results_pose = pose.process(image_rgb)

    # Draw face landmarks on the image
    if results_face_mesh.multi_face_landmarks:
        face_landmarks = results_face_mesh.multi_face_landmarks[0]  # Consider only the first detected face
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
        )

        # Get the center of the face
        face_center = (
            int(face_landmarks.landmark[0].x * image.shape[1]),
            int(face_landmarks.landmark[0].y * image.shape[0])
        )

        # Calculate the convex hull of the face landmarks
        points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in face_landmarks.landmark]
        hull = cv2.convexHull(np.array(points, dtype=np.float32), returnPoints=True)

        # Calculate the area of the convex hull
        face_area = cv2.contourArea(hull)
        
        # Assume shoulder landmark is known (you may need to change the index based on your use case)
        shoulder_landmark_index = 12  # Example: assuming shoulder landmark is index 12

        # Get the shoulder landmark position
        shoulder_landmark = face_landmarks.landmark[shoulder_landmark_index]

        # Assume face landmark for angle calculation (you may need to change the index based on your use case)
        face_landmark_index = 168  # Example: assuming face landmark is index 168

        # Get the face landmark position
        face_landmark = face_landmarks.landmark[face_landmark_index]

        # Calculate the upward angle of the face with respect to the shoulder
        angle1 = calculate_face_angle(shoulder_landmark, face_landmark)

    # Draw shoulder landmarks on the image
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        # Get the left shoulder coordinates
        left_shoulder = (
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])
        )

        # Get the right shoulder coordinates
        right_shoulder = (
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])
        )
        
        left_shoulder_z = (results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                         results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z)
        right_shoulder_z = (results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                          results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
        
        # Calculate the distance between the face center and the shoulder
        distance = calculate_distance(face_center, [(left_shoulder[0] + right_shoulder[0]) // 2,(left_shoulder[1] + right_shoulder[1]) // 2])

        # Calculate the shoulder width as the distance between the left and right shoulder
        shoulder_width = math.dist(left_shoulder, right_shoulder)

        # Calculate the relative distance
        relative_distance = calculate_relative_distance(distance, shoulder_width)

        # Calculate the angle of the shoulder with respect to the image frame
        shoulder_angle = calculate_shoulder_angle_y(left_shoulder, right_shoulder)

        # Calculate the ratio between the face area and shoulder width
        ratio = face_area / shoulder_width
        diff = right_shoulder_z[2] - left_shoulder_z[2]

        # #     ## comment these out if don't want camera display
        sdf = an = ra = sa = rd = False

        # Display a green circle in the top-left corner if the conditions  are met
        if relative_distance > RDV:
            rd = True
        if (shoulder_angle < -175 or shoulder_angle > 175):
            sa = True
        if ratio < RAT:
            ra = True
        if angle1 < ANG2 and angle1 > ANG1:
            an = True
        if diff < 0.2:
            sdf = True
        cv2.putText(image, f"Relative Distance: {rd}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, f"Shoulder Angle: {sa}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Ratio: {ra}, ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"face upward angle: {an}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, f"shoulder z diff: {sdf}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

        if rd and sa and ra and an and sdf:
            cv2.circle(image, (600, 50), 25, (0, 255, 0), -1)
        else:
            cv2.circle(image, (600, 50), 50, (0, 125, 255), -1)
            # os.system('say "Sloucher"')
            # os.system('afplay /System/Library/Sounds/Sosumi.aiff')
            # break

        # Show the resulting image
        cv2.imshow('Face and Shoulder Measurements', image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_count % frame_skip_interval != 0:
            continue
        # Display a notification in the taskbar if the conditions are met
        if (
            relative_distance > RDV
            and (shoulder_angle < -175 or shoulder_angle > 175)
            and ratio < RAT
            and (angle1 < ANG2 and angle1 > ANG1)
            and diff < 0.2
        ):
            status = 'Good posture'

            if status != pre_status or status is None:
                print(status)
                notification.notify(
                    title="Good posture",
                    message="Please keep!",
                    timeout=3  # Notification will be displayed for 3 seconds
                )
                pre_status = 'Good posture'
        elif (relative_distance <= RDV):
            status = 'Slouching'
            if status != pre_status or status is None:
                print(status)
                notification.notify(
                    title="Bad posture",
                    message="You are slouching, keep your back straight!",
                    timeout=3  # Notification will be displayed for 3 seconds
                )
                pre_status = 'Slouching'
        elif (shoulder_angle >= -175 and shoulder_angle <= 175):
            status = 'Lopsided shoulders'
            if status != pre_status or status is None:
                print(status)
                notification.notify(
                    title="Bad posture",
                    message="Your shoulders are lopsided, please adjust them!",
                    timeout=3  # Notification will be displayed for 3 seconds
                )
                pre_status = 'Lopsided shoulders'
        elif (ratio >= RAT):
            status = 'Too close'
            if status != pre_status or status is None:
                print(status)
                notification.notify(
                    title="Bad posture",
                    message="Your eyes are too close to the computer, please move farther away!",
                    timeout=3  # Notification will be displayed for 3 seconds
                )
                pre_status = 'Too close'
        elif (angle1 >= ANG2):
            status = 'facing up'
            if status != pre_status or status is None:
                print(status)
                notification.notify(
                    title="Bad posture",
                    message="Please lower your head!",
                    timeout=3  # Notification will be displayed for 3 seconds
                )
                pre_status = 'facing up'
        elif (angle1 <= ANG1):
            status = 'facing down'
            if status != pre_status or status is None:
                print(status)
                notification.notify(
                    title="Bad posture",
                    message="Please raise your head!",
                    timeout=3  # Notification will be displayed for 3 seconds
                )
                pre_status = 'facing down'
        elif (diff >= 0.2):
            status = 'shoulders in tandem'
            if status != pre_status or status is None:
                print(status)
                notification.notify(
                    title="Bad posture",
                    message="Your shoulders are one front and the other back, please adjust them!",
                    timeout=3  # Notification will be displayed for 3 seconds
                )
                pre_status = 'shoulders in tandem'
                
# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
