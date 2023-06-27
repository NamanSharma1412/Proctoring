from streamlit_webrtc import webrtc_streamer
from eyeTracking_headTracking import euclidean_dist,analyze_frame
import math
import numpy as np
import av
import cv2
import imutils
import numpy as np
import base64
import time
import socket
import mediapipe as mp
import itertools
import math

class VideoProcessor: 
    
    def recv(self,frame):
        frm = frame.to_ndarray(format = 'bgr24')
        # cv2.putText(frm,'HII',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
        frm = analyze_frame(frm)
        return av.VideoFrame.from_ndarray(frm,format = 'bgr24')

webrtc_streamer(key = 'KEY',video_processor_factory=VideoProcessor)