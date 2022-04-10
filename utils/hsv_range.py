import cv2
import numpy as np
from sys import argv


def adjust_hsv(image=None):
    # optional argument for trackbars
    def nothing(x):
        pass

    # named ites for easy reference
    barsWindow = 'Bars'

    hl, hh = 'H Low', 'H High'
    sl, sh = 'S Low', 'S High'
    vl, vh = 'V Low', 'V High'

    # create window for the slidebars
    cv2.namedWindow(barsWindow)
    cv2.resizeWindow(barsWindow, 400, 200)

    # create the sliders
    cv2.createTrackbar(hl, barsWindow, 0, 179, nothing)
    cv2.createTrackbar(hh, barsWindow, 0, 179, nothing)
    cv2.createTrackbar(sl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(sh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(vl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(vh, barsWindow, 0, 255, nothing)

    # set initial values for sliders
    cv2.setTrackbarPos(hl, barsWindow, 0)
    cv2.setTrackbarPos(hh, barsWindow, 179)
    cv2.setTrackbarPos(sl, barsWindow, 0)
    cv2.setTrackbarPos(sh, barsWindow, 255)
    cv2.setTrackbarPos(vl, barsWindow, 0)
    cv2.setTrackbarPos(vh, barsWindow, 255)


    use_camera = False

    if image is None :
        cap = cv2.VideoCapture(0)
        use_camera = True
    elif isinstance(image, str):
        try: 
            frame = cv2.imread(image)
        except:
            print('Error reading image')
            exit()
    else : 
        frame = image

    cv2.namedWindow("Masked")
    cv2.resizeWindow("Masked", 400, 200)

    while(cv2.getWindowProperty('Masked', 0) >= 0):

        if use_camera:
            ret, frame = cap.read()
        
        # convert to HSV from BGR
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # read trackbar positions for all
        hul = cv2.getTrackbarPos(hl, barsWindow)
        huh = cv2.getTrackbarPos(hh, barsWindow)
        sal = cv2.getTrackbarPos(sl, barsWindow)
        sah = cv2.getTrackbarPos(sh, barsWindow)
        val = cv2.getTrackbarPos(vl, barsWindow)
        vah = cv2.getTrackbarPos(vh, barsWindow)

        # make array for final values
        HSVLOW = np.array([hul, sal, val])
        HSVHIGH = np.array([huh, sah, vah])

        # apply the range on a mask
        mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
        maskedFrame = cv2.bitwise_and(frame, frame, mask = mask)

        # display the camera and masked images
        cv2.imshow('Masked', maskedFrame)
        # cv2.imshow('Camera', frame)

        # check for q to quit program with 5ms delay
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # clean up our resources
    if use_camera:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    if len(argv) == 2: 
        adjust_hsv(argv[1])
    else:
        adjust_hsv()