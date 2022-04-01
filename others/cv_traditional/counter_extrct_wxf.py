import cv2

imgpath = '/home/dell/下载/webwxgetmsgimg(1).jpeg'

img = cv2.imread(imgpath)
h,w,c = img.shape
# img = img[1400:2200,:,:]
cv2.imshow('img', img)
# cv2.waitKey()
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('img', img_gray)
cv2.waitKey()

for thresh in [50, 210, 240]:
    img_b = cv2.threshold(img_gray, thresh, 255,  cv2.THRESH_BINARY)[1]
    cv2.imshow('img', img_b)
    # cv2.waitKey()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('img', img_b)
    # cv2.waitKey()

    contours, hierarchy = cv2.findContours(img_b,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key= lambda x:x.shape[0])
    img = cv2.drawContours(img,contours[-1],-1,(0,0,255),1)
    cv2.imshow('img', img)
    # cv2.waitKey()

cv2.imwrite('wxf.png', img)

