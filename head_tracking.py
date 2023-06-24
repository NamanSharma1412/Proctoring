import cv2
import mediapipe as mp
import numpy as np
import itertools
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

cap = cv2.VideoCapture("C:\\Users\\naman\\Pictures\\Camera Roll\\WIN_20230615_00_10_22_Pro.mp4")
NOSE = 1
RIGHT_EYE = 159
LEFT_EYE = 386
FOREHEAD = 151
RIGHT_CHEEK = 101
LEFT_CHEEK = 130
CHIN = 199
MOUTH = 0
req_index = [1,159,151,386,0,199,101,130]
while cap.isOpened():
    face_2d = []
    face_3d = []
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,refine_landmarks = True,min_tracking_confidence=0.5) as face_mesh:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame,1)
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_h,img_w = img.shape[:2]
            img.flags.writeable = False
            results = face_mesh.process(img)
            img.flags.writeable = True
            if results.multi_face_landmarks:
                for face_landmark in results.multi_face_landmarks:
                    for idx,p in enumerate(face_landmark.landmark):
                        if idx in req_index:
                            x,y = p.x*img_w,p.y*img_h
                            face_2d.append([x,y])
                            z = p.z
                            face_3d.append([x,y,z])
#                 face_2d.append([mesh_points[NOSE][0],mesh_points[NOSE][1]])
#                 face_2d.append([mesh_points[CHIN][0],mesh_points[CHIN][1]])
#                 face_2d.append([mesh_points[LEFT_EAR][0],mesh_points[LEFT_EAR][1]])
#                 face_2d.append([mesh_points[LEFT_EYE][0],mesh_points[LEFT_EYE][1]])
#                 face_2d.append([mesh_points[RIGHT_EAR][0],mesh_points[RIGHT_EAR][1]])
#                 face_2d.append([mesh_points[RIGHT_EYE][0],mesh_points[RIGHT_EYE][1]])
                
#                 face_3d.append([mesh_points[NOSE][0],mesh_points[NOSE][1],mesh_points[NOSE][2]])
#                 face_3d.append([mesh_points[CHIN][0],mesh_points[CHIN][1],mesh_points[CHIN][2]])
#                 face_3d.append([mesh_points[LEFT_EAR][0],mesh_points[LEFT_EAR][1],mesh_points[LEFT_EAR][2]])
#                 face_3d.append([mesh_points[LEFT_EYE][0],mesh_points[LEFT_EYE][1],mesh_points[LEFT_EYE][2]])
#                 face_3d.append([mesh_points[RIGHT_EAR][0],mesh_points[RIGHT_EAR][1],mesh_points[RIGHT_EAR][2]])
#                 face_3d.append([mesh_points[RIGHT_EYE][0],mesh_points[RIGHT_EYE][1],mesh_points[RIGHT_EYE][2]])
                
                    face_2d = np.array(face_2d,dtype = np.float64)
                    face_3d = np.array(face_3d,dtype = np.float64)

                    focal_length = 1*img_w
                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    if y < -10:
                        text = "Looking Left"
                    elif y > 10:
                        text = "Looking Right"
                    elif x < -10:
                        text = "Looking Down"
                    elif x > 10:
                        text = "Looking Up"
                    else:
                        text = "Forward"

                    cv2.putText(frame,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)    
         
        
                mp_drawing.draw_landmarks(
                        image = frame,
                        landmark_list = face_landmark,
                        connections = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec = drawing_spec
                    )
        
        cv2.imshow('image',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()