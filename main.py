from tkinter import NW
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import time
from PIL.ImageTk import PhotoImage
from call import call
import tkinter.ttk as ttk
import threading
import datetime
import imutils

changes = "1.0:\nTrack released! You can choose to play with or without a camera.\n\n1.1:\nVideo calling! You can now " \
          "host a call\nbetween two computers on the same network.\nThe video from " \
          "the host will be sent to the client.\n\n1.2:\nSmall UI changes, making the whole experience nicer. "

called = 0
cam = 0
change = 0
changeCalled = 0


def camShow():
    global cam
    global called

    called += 1

    if called % 2 == 0:
        cam = 0
    elif called % 2 != 0:
        cam = 1

    if cam == 1:
        btn_showHideCamera.config(image=photoNoCam)
    elif cam == 0:
        btn_showHideCamera.config(image=photoCam)


def changeLog():
    global change
    global changeCalled

    changeCalled += 1

    if changeCalled % 2 == 0:
        change = 0
    elif changeCalled % 2 != 0:
        change = 1

    if change == 1:
        labelDescription.config(text=changes)
    elif change == 0:
        labelDescription.config(text="JKinc and Squiggle Development present: Track!\n\nTrack is a simple, "
                                     "easy body tracking program,\nwhere every movement you do will show up on "
                                     "your Line Avatar.\nTo close, simply press the escape key on your "
                                     "keyboard.")


def play():
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
                    widthFloat = cap.get(3)
                    heightFloat = cap.get(4)
                    width = int(widthFloat)
                    height = int(heightFloat)
                    # print(width) for debugging; not required
                    # print(height)
                    if cam == 1:
                        image = np.zeros((height, width, 3), np.uint8)

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
                    overlayImg = cv2.imread("media/TrackSmallFlipped.png")

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

                    root = tk.Tk(className="Track")
                    root.iconbitmap("media/icon.ico")

                    panel = tk.Label(image=image)
                    panel.image = image

                    if panel is None:
                        panel = tk.Label(image=image)
                        panel.image = image
                        panel.pack(side="left", padx=10, pady=10)

                        # otherwise, simply update the panel
                    else:
                        panel.configure(image=image)
                        panel.image = image

    cap.release()
    cv2.destroyAllWindows()


window = tk.Tk(className=" Track Launcher")
window.iconbitmap("media/icon.ico")

photoCam = PhotoImage(file=r"media/Camera.png")
photoLink = PhotoImage(file=r"media/Link.png")
photoNoCam = PhotoImage(file=r"media/NoCamera.png")

window.configure(bg="#2C2C2C")

img = Image.open("media/TrackSmall.png")
imgDisplay = ImageTk.PhotoImage(img)

btn_joinHostCall = tk.Button(text="Join/Host\nVideo Call", command=call)
btn_showHideCamera = tk.Button(text="Camera:\nShowing", command=camShow)
btn_play = tk.Button(text="Play!", command=play)
btn_changelog = tk.Button(text="Change\nLog", command=changeLog)

labelImage = tk.Label(image=imgDisplay)
labelImage.image = imgDisplay
labelImage.config(bg="#2C2C2C")

labelDescription = tk.Label(window, text="JKinc and Squiggle Development present: Track!\n\nTrack is a simple, "
                                         "easy body tracking program,\nwhere every movement you do will show up on "
                                         "your Line Avatar.\nTo close, simply press the escape key on your "
                                         "keyboard.")

labelDescription.config(bg="#2C2C2C", fg="white", font=("Century Gothic", 12))

btn_joinHostCall.config(bg="#2C2C2C", fg="white", image=photoLink, width=75, height=50)
btn_play.config(bg="#2C2C2C", fg="white", font=("Century Gothic Bold", 16))
btn_showHideCamera.config(bg="#2C2C2C", fg="white", image=photoCam, width=75, height=50)
btn_changelog.config(bg="#2C2C2C", fg="white", width=6, height=2, font=("Century Gothic", 9))

labelImage.grid(row=0, column=1)
labelDescription.grid(row=1, column=1, sticky="nsew")
btn_joinHostCall.grid(row=2, column=0, sticky="nsew")
btn_play.grid(row=2, column=1, sticky="nsew")
btn_showHideCamera.grid(row=2, column=2)
btn_changelog.grid(row=0, column=0)


def func(event):
    print("Just something.")


def onclick(event=None):
    labelDescription.config(text="Wow! Well done! You pressed enter and achieved:\n\n\nNothing.")


window.bind("<Return>", onclick)

window.mainloop()
