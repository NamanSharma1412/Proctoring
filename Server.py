import cv2
import imutils
import numpy as np
import base64
import time
import socket

BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name) 
# print(host_ip)                        # 192.168.196.108
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
print(f"Listening at {socket_address}")
cap = cv2.VideoCapture(0)   # Video Data
fps,st,frames_to_count,cnt = (0,0,20,0)
while True:
    message,client_addr = server_socket.recvfrom(BUFF_SIZE)
    print('GOT CONNECTION FROM ',client_addr)
    WIDTH = 400
    while (cap.isOpened()):
        ret,frame = cap.read()
        if ret:
            frame = imutils.resize(frame,width = WIDTH)
            encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
            message = base64.b64encode(buffer)
            server_socket.sendto(message,client_addr)
            cv2.imshow('TRANSMITTED VIDEO',frame)
            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                server_socket.close()
                cap.release()
                cv2.destroyAllWindows()
                break