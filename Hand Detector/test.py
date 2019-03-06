'''

Air Writing - Translating gestures to on-screen handwriting

'''


from hand_recognition import *
import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# initialize a black canvas
screen = np.zeros((600, 1000))


# use this to capture a live histogram
hist = capture_histogram(0)


# to save the histogram locally, in the same directory
# np.save('hist.npy', hist)


# to load a saved histogram
# hist = np.load('hist.npy')


curr = None
prev = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flip and resize the image.
    frame = cv2.flip(frame, 1)
    # Use a resolution best suited for your camera.
    frame = cv2.resize(frame, (1000, 600))

    hand_detected, hand = detect_hand(frame, hist)

    if hand_detected:
        hand_image = hand["boundaries"]

        fingertips = extract_fingertips(hand)
        plot(hand_image, fingertips)

        prev = curr
        curr = fingertips[0]

        # draw a line on the black screen from previous fingertip location
        # to current fingertip location
        if prev and curr:
            cv2.line(screen, prev, curr, (255, 255, 255), 5)

        cv2.imshow("Drawing", screen)
        cv2.imshow("Hand Detector", hand_image)

    else:
        cv2.imshow("Hand Detector", frame)

    k = cv2.waitKey(5)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
