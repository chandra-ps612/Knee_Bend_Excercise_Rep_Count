import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils # It'll use to get all the drawing utilities from MediaPipe python package
mp_pose = mp.solutions.pose # It is used to import the pose estimation model
#mp_holistic = mp.solutions.holistic
videoPath='C:\\Users\\lav singh\\AppData\\Local\\Programs\\Python\\Python39\\exp_1\\Knee_Bend_Exercise_Rep_Count\\KneeBend.mp4'
output_videoPath='C:\\Users\\lav singh\\AppData\\Local\\Programs\\Python\\Python39\\exp_1\\Knee_Bend_Exercise_Rep_Count\\output.mp4'
Rep_count= 0
time_counter=0
timer_limit=8
create= None
def leg_landmarks_position(frame, draw=True):
  h, w, c = frame.shape
  ll = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  rl = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  if(results.pose_landmarks):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    for id, ma in enumerate(results.pose_landmarks.landmark):
      if(id == 23 and ma.visibility > 0.1):
        ll[0][0] = int(ma.x*w)
        ll[0][1] = int(ma.y*h)
        #ll[0][2] = int(ma.z*w)
      if(id == 25 and ma.visibility > 0.1):
        ll[1][0] = int(ma.x*w)
        ll[1][1] = int(ma.y*h)
        #ll[1][2] = int(ma.z*w)
      if(id == 27 and ma.visibility > 0.1):
        ll[2][0] = int(ma.x*w)
        ll[2][1] = int(ma.y*h)
        #ll[2][2] = int(ma.z*w)
      if(id == 24 and ma.visibility > 0.1):
        rl[0][0] = int(ma.x*w)
        rl[0][1] = int(ma.y*h)
        #rl[0][2] = int(ma.z*w)
      if(id == 26 and ma.visibility > 0.1):
        rl[1][0] = int(ma.x*w)
        rl[1][1] = int(ma.y*h)
        #rl[1][2] = int(ma.z*w)
      if(id == 28 and ma.visibility > 0.1):
        rl[2][0] = int(ma.x*w)
        rl[2][1] = int(ma.y*h)
        #rl[2][2] = int(ma.z*w)
  return ll, rl
def get_angle(list):
  a = np.array(list[0])
  b = np.array(list[1])
  c = np.array(list[2])
  ba = a-b
  bc = c-b
  cosine_angle= np.dot(ba, bc)/(np.linalg.norm(ba) * np.linalg.norm(bc))
  angle =np.arccos(cosine_angle).astype(int) 
  return int(np.degrees(angle))

# Video source:
cap = cv2.VideoCapture(videoPath)
# Setting up mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, frame=cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Mediapipe works with RGB image
    # Passing the RGB image through pose estimation model to make the detections
    results = pose.process(frame)
    # Draw the pose annotation on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Converting image into BGR for OpenCV
    ll, lr = leg_landmarks_position(frame, draw=True)
    left_leg_angle= get_angle(ll)
    right_leg_angle= get_angle(lr)
    try: # For avoiding repetations of frame 
      if left_leg_angle==171 or right_leg_angle==171:
        cv2.putText(frame, f'Keep your knee bent', (200, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1 )
      elif (57<=left_leg_angle<=114) or (57<=right_leg_angle<=114):
        current_time=time.time() # Storing current time
      if current_time-time_counter < timer_limit+1: # Starting Countdown_Timer
        cv2.rectangle(frame, (510, 5), (640, 30), (255, 165, 0), -1)
        cv2.putText(frame, f'Timer_Count:{str(int(timer_limit+1-(current_time-time_counter)))}', (510, 20),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
      elif (current_time-time_counter) > timer_limit:
        time_counter=current_time # Reset Timer_Count to zero
        print(f'Relax')
        Rep_count+=1
    except:
      pass
    cv2.rectangle(frame, (8, 5), (500, 30), (255, 255, 0), -1)
    cv2.putText(frame, f'left_leg_angle:{get_angle(ll)} | right_leg_angle:{get_angle(lr)} | Rep_count:{Rep_count}', (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow('Knee_bend_excercise', frame)
    if create is None:
      fourcc = cv2.VideoWriter_fourcc(*'MP4V')
      create = cv2.VideoWriter(output_videoPath, fourcc, 30, (640, 480), True)
    create.write(frame)
    
    if cv2.waitKey(1)==ord('q'):
      break
cap.release()
create.release()
cv2.distroyAllWindows()