import cv2

import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny = cv2.Canny(blur_img, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height),
                          (600, 280)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def hough_lines(image):
    lines = cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=10)
    return lines
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(int(x1),int(y1)) , (int(x2),int(y2)),(255,0,0) , 8)
    return  line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_cordinates(image, left_fit_average)
    right_line = make_cordinates(image, right_fit_average)
    return np.array([left_line, right_line])
def make_cordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = image.shape[0]
    y2 = int(y1*(0.5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])



cap = cv2.VideoCapture("test_drive.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    lane_img = np.copy(frame)

    canny_img = canny(lane_img)
    cropped_img = region_of_interest(canny_img)

    lines = hough_lines(cropped_img)
    averaged_lines = average_slope_intercept(lane_img, lines)
    line_img = display_lines(lane_img, averaged_lines)
    combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)


    cv2.imshow("image", combo_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllWindows()


