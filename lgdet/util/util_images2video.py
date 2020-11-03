import cv2
import os

# 图片路径
im_dir = 'F:/Projects/auto_Airplane/TS02/truck_tracker/tracker'
# im_dir = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/000'
# 输出视频路径
video_dir = 'util_tmp/video.avi'
# 帧率
fps = 24

files = os.listdir(im_dir)


# 图片尺寸
img_size = cv2.imread(os.path.join(im_dir,files[0])).shape[:2][::-1]

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)


for files in files:
    im_name = os.path.join(im_dir, files)
    frame = cv2.imread(im_name)
    video_writer.write(frame)
    cv2.imshow('rr', frame)
    cv2.waitKey(1)

video_writer.release()
print('finish')
