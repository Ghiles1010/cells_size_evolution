import cv2

def crop_image_gui(image):
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    oriImage = image.copy()

    def mouse_crop(event, x, y, flags, param):
        nonlocal x_start, y_start, x_end, y_end, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y
        
        elif event == cv2.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False

            roi = oriImage[y_start:y_end, x_start:x_end]

            h, w, _ = roi.shape
            if h < 50 or w < 50:
                roi = cv2.resize(roi, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

            cv2.imshow("Cropped", roi)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while cv2.getWindowProperty('image', 0) >= 0:
        img_cpy = image.copy()
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.rectangle(img_cpy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
            cv2.imshow("image", img_cpy)
        cv2.waitKey(1)

    cv2.destroyWindow('image')
    cv2.destroyWindow('Cropped')

    return y_start, y_end, x_start, x_end