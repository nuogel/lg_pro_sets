from moviepy.editor import VideoFileClip, concatenate_videoclips


filepath='/home/luogeng/ssd_datasets/datasets/uav/对比/效果差的视频.mp4'
clip2 = VideoFileClip(filepath).subclip('00:08:23','00:13:00')

fianl = concatenate_videoclips([clip2])
fianl.write_videofile('cut_out.mp4')
