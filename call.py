from time import sleep
from tkinter import NW, Frame
import socket
import cv2
import mediapipe as mp
from time import sleep
import numpy as np
import time
import tkinter as tk
import threading
from PIL import ImageTk, Image
import tkinter.ttk as ttk

from PIL.ImageTk import PhotoImage


def host():

    tk.messagebox.showinfo(title="Host Track", message="Waiting For Client Connection")

    HOST = ''  # Symbolic name meaning all available interfaces
    PORT = 50007  # Arbitrary non-privileged port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    pTime = 0
    NUM_FACE = 2
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    mp_holistic = mp.solutions.holistic

    # For webcam input:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            with mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    start_time = time.time()
                    success, image = cap.read()
                    org = image
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    results2 = hands.process(image)
                    results3 = holistic.process(image)
                    image = np.zeros((1000, 1000, 3), np.uint8)
                    if results2.multi_hand_landmarks:
                        for hand_landmarks in results2.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    mp_drawing.draw_landmarks(
                        image,
                        results3.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style()
                    )
                    overlayImg = cv2.imread("TrackSmallFlipped.png")

                    # I want to put logo on top-left corner, So I create a ROI
                    rows, cols, channels = overlayImg.shape

                    roi = image[0:rows, 0:cols]

                    # Now create a mask of logo and create its inverse mask also
                    img2gray = cv2.cvtColor(overlayImg, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # Now black-out the area of logo in ROI
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    # Take only region of logo from logo image.
                    img2_fg = cv2.bitwise_and(overlayImg, overlayImg, mask=mask)

                    # Put logo in ROI and modify the main image
                    dst = cv2.add(img1_bg, img2_fg)
                    image[0:rows + 0, 0:cols + 0] = dst

                    # Flip the image horizontally for a selfie-view display.
                    frame = cv2.flip(image, 1)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),90]
                    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                    data = np.array(imgencode)
                    stringData = data.tobytes()
                    conn.send(stringData)
                    if cv2.waitKey(1) == 27:
                        break
                    print(1.0 / (time.time() - start_time))

    cap.release()


def join_receive(sock):
    while True:
        try:
            data = sock.recv(2097152)
            data = np.frombuffer(data, dtype='uint8')

            frame = cv2.imdecode(data, 1)
            cv2.imshow("Track", frame)
            if cv2.waitKey(1) == 27:
                break
        except:
            pass
    input


def join(host):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host, 50007)
    sock.connect(server_address)
    receive = threading.Thread(target=join_receive, args=(sock,))
    receive.start()


def call():
    root = tk.Tk()
    root.geometry("240x100")
    root.title(" Join/Host")
    root.resizable(0, 0)
    root.iconbitmap("media/icon.ico")

    # Configure the grid
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)

    # Session ID
    sessionID_label = ttk.Label(root, text="Session ID:")
    sessionID_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

    sessionID_entry = ttk.Entry(root)
    sessionID_entry.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

    # Join button
    join_button = ttk.Button(root, text="Join", command=lambda: join(sessionID_entry.get()))
    join_button.grid(column=1, row=2, sticky=tk.E, padx=5, pady=5)

    # Host button
    host_button = ttk.Button(root, text="Host", command=host)
    host_button.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

    root.mainloop()


