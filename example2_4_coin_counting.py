#Download images from https://drive.google.com/file/d/1KqllafwQiJR-Ronos3N-AHNfnoBb8I7H/view?usp=sharing

import cv2
import numpy as np

def coinCounting(filename):
    im = cv2.imread(filename)
    target_size = (500, 500)
    im = cv2.resize(im,target_size)
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(im_g, (51,51), 0)
    clip = np.clip(blur, 1, 255)
    merge = cv2.merge([clip] * 3)
    background = np.clip(((im/merge)*125), 1, 255).astype(np.uint8)
    background_b = np.clip(((im/merge)*190), 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(background, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 130, 90])
    yellow_upper = np.array([30, 255, 255])
    hsv_erode_yellow = cv2.erode(hsv, np.ones((5, 5), np.uint8))
    yellow_mask = cv2.inRange(hsv_erode_yellow, yellow_lower, yellow_upper)
    blur_yellow = cv2.medianBlur(yellow_mask, 3)
    yellow = cv2.morphologyEx(blur_yellow, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    hsv_erode_blue = cv2.erode(background_b, np.ones((22, 22), np.uint8))
    hsv_dilate_blue = cv2.dilate(hsv_erode_blue, np.ones((5, 1), np.uint8))
    blue_mask = cv2.inRange(hsv_dilate_blue,(180,145,0),(255,255,150))
    blue = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    blue = cv2.erode(blue, np.flipud(np.eye(7, dtype=np.uint8)))

    blue_contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    yellow_contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    blue_count = len(blue_contours)
    yellow_count = len(yellow_contours)
    # print(f"Blue objects: {blue_count}")
    # print(f"Yellow objects: {yellow_count}")

    cv2.putText(
        im,
        f"[yellow:{yellow_count} , blue:{blue_count}]",
        (0, 100),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=3,
        thickness=3,
        color=(255, 255, 0),
    )

    cv2.imshow('ori', im)
    cv2.moveWindow('ori',10,10)
    # cv2.imshow('blue', blue_mask)
    # cv2.moveWindow('blue',1200,10)
    # cv2.imshow('yellow', yellow)
    # cv2.moveWindow('yellow',600,10)
    cv2.waitKey()

    return [yellow_count,blue_count]

for i in range(1,11):
    print(i,":",coinCounting('CoinCounting/coin'+str(i)+'.jpg'))
