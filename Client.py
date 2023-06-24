import cv2
import imutils
import numpy as np
import base64
import time
import socket
import mediapipe as mp
import itertools
import math
from eyeTracking_headTracking import analyze_frame,euclidean_dist # imported from eyeTracking_headTracking.py

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)


BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '192.168.0.110'
port = 9990
host_addr = (host_ip,port)
message = b'hello'
client_socket.sendto(message,host_addr)

while True:
    packet,_ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet,' /')
    npdata = np.fromstring(data,dtype=np.uint8)
    frame = cv2.imdecode(npdata,1)
    analyzed_frame = analyze_frame(frame)
    cv2.imshow('RECEIVED ANALYZED VIDEO',analyzed_frame)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        client_socket.close()
        cv2.destroyAllWindows()
        break

    