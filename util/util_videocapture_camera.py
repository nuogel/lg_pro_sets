import cv2


def get_camera_image():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        print('reading')
        cv2.imshow('img', frame)
        cv2.waitKey()

    # return frame


if __name__ == '__main__':
    get_camera_image()