def euclidean_dist(p1,p2):
    import math
    import numpy as np
    x1,y1 = np.ravel(p1)
    x2,y2 = np.ravel(p2)
    return int((math.sqrt(((x2-x1)**2+(y2-y1)**2))))

def analyze_frame(frame):        # input frame of stream video, outputs analyzed frame with facemesh drawing
    import cv2
    import imutils
    import numpy as np
    import base64
    import time
    import socket
    import mediapipe as mp
    import itertools
    import math
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1) 
    LEFT_IRIS_INDICES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_IRIS)))
    RIGHT_IRIS_INDICES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_IRIS)))
    RIGHT_EYE_RIGHTMOST = [33]
    RIGHT_EYE_LEFTMOST = [133]
    LEFT_EYE_RIGHTMOST = [362]
    LEFT_EYE_LEFTMOST = [263]
    LEFT_EYE_TOP = [443]
    LEFT_EYE_BOTTOM = [450]
    RIGHT_EYE_TOP = [223]
    RIGHT_EYE_BOTTOM = [230]
    NOSE = 1
    RIGHT_EYE = 159
    LEFT_EYE = 386
    FOREHEAD = 151
    RIGHT_CHEEK = 101
    LEFT_CHEEK = 130
    CHIN = 199
    MOUTH = 0
    req_index = [1,159,151,386,0,199,101,130]

    
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,refine_landmarks = True,min_tracking_confidence=0.5) as face_mesh:
        face_2d = []
        face_3d = []
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = face_mesh.process(img)
        img.flags.writeable = True
        if results.multi_face_landmarks:
            img_h,img_w = img.shape[:2]
            mesh_points = np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    #         print(mesh_points.shape)
    #         cv2.polylines(frame,[mesh_points[LEFT_EYE]],True,(0,255,0),2,cv2.LINE_AA)
    #         cv2.polylines(frame,[mesh_points[RIGHT_EYE]],True,(0,255,0),2,cv2.LINE_AA)
    #         cv2.polylines(frame,[mesh_points[LEFT_IRIS_INDICES]],True,(10,255,0),1,cv2.LINE_AA)
    #         cv2.polylines(frame,[mesh_points[RIGHT_IRIS_INDICES]],True,(10,255,0),1,cv2.LINE_AA)
            (l_cx,l_cy),l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS_INDICES])
            (r_cx,r_cy),r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS_INDICES])

            center_left = np.array([l_cx,l_cy],dtype = int)
            center_right = np.array([r_cx,r_cy],dtype = int)
            left_left = np.array(mesh_points[LEFT_EYE_LEFTMOST],dtype = int)
            left_right = np.array(mesh_points[LEFT_EYE_RIGHTMOST],dtype = int)
            right_right = np.array(mesh_points[RIGHT_EYE_RIGHTMOST],dtype = int)
            right_left = np.array(mesh_points[RIGHT_EYE_LEFTMOST],dtype = int)
            right_top = np.array(mesh_points[RIGHT_EYE_TOP],dtype = int)
            right_bottom = np.array(mesh_points[RIGHT_EYE_BOTTOM],dtype = int)
            left_top = np.array(mesh_points[LEFT_EYE_TOP],dtype = int)
            left_bottom = np.array(mesh_points[LEFT_EYE_BOTTOM],dtype = int)

            cv2.circle(frame,center_left,int(l_radius),(255,0,255),1,cv2.LINE_AA)
            cv2.circle(frame,center_right,int(r_radius),(255,0,255),1,cv2.LINE_AA)

            cv2.circle(frame,center_left,1,(255,0,255),1,cv2.LINE_AA)
            cv2.circle(frame,center_right,1,(255,0,255),1,cv2.LINE_AA)

            cv2.circle(frame,(left_left[0][0],left_left[0][1]),1,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame,(left_right[0][0],left_right[0][1]),1,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame,(right_left[0][0],right_left[0][1]),1,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame,(right_right[0][0],right_right[0][1]),1,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame,(left_top[0][0],left_top[0][1]),1,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame,(left_bottom[0][0],left_bottom[0][1]),1,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame,(right_top[0][0],right_top[0][1]),1,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame,(right_bottom[0][0],right_bottom[0][1]),1,(0,255,0),1,cv2.LINE_AA)

            dist_left_left = euclidean_dist(center_left,left_left)/euclidean_dist(left_left,left_right)
            dist_left_right = euclidean_dist(center_left,left_right)/euclidean_dist(left_left,left_right)
            dist_right_left = euclidean_dist(center_right,right_left)/euclidean_dist(right_left,right_right)
            dist_right_right = euclidean_dist(center_right,right_right)/euclidean_dist(right_left,right_right)

            dist_right_top = euclidean_dist(center_right,right_top)/euclidean_dist(right_top,right_bottom)
            dist_right_bottom = euclidean_dist(center_right,right_bottom)/euclidean_dist(right_top,right_bottom)
            dist_left_top = euclidean_dist(center_left,left_top)/euclidean_dist(left_top,left_bottom)
            dist_left_bottom = euclidean_dist(center_left,left_bottom)/euclidean_dist(left_top,left_bottom)

            cv2.line(frame, (left_left[0][0], left_left[0][1]), (left_right[0][0], left_right[0][1]), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (right_left[0][0], right_left[0][1]), (right_right[0][0], right_right[0][1]), (0, 255, 0), 1, cv2.LINE_AA)

            cv2.line(frame, (left_top[0][0], left_top[0][1]), (left_bottom[0][0], left_bottom[0][1]), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (right_top[0][0], right_top[0][1]), (right_top[0][0], right_bottom[0][1]), (0, 255, 0), 1, cv2.LINE_AA)

    #             cv2.putText(frame,str(dist_left_left),left_left[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)
    #             cv2.putText(frame,str(dist_left_right),left_right[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)
    #             cv2.putText(frame,str(dist_right_left),right_left[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)
    #             cv2.putText(frame,str(dist_right_right),right_right[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)

    #             cv2.putText(frame,str(dist_left_top),left_top[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)
    #             cv2.putText(frame,str(dist_left_bottom),left_bottom[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)
    #             cv2.putText(frame,str(dist_right_top),right_top[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)
    #             cv2.putText(frame,str(dist_right_bottom),right_bottom[0],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1,cv2.LINE_AA)

            if(dist_left_left>dist_left_right and dist_right_left>dist_right_right):
                cv2.putText(frame,'right',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)

            elif(dist_right_left<dist_right_right and dist_left_left<dist_left_right):
                cv2.putText(frame,'left',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)

            elif(dist_left_top>dist_left_bottom+0.1 and dist_right_top>dist_right_bottom+0.1):
                cv2.putText(frame,'down',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
            elif(dist_left_top+0.1<dist_left_bottom and dist_right_top+0.1<dist_right_bottom):
                cv2.putText(frame,'up',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)

            else:
                cv2.putText(frame,'center',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
            
            
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

                cv2.putText(frame,text,(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1) 
                
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmark,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec = drawing_spec
            )

            

    return frame
