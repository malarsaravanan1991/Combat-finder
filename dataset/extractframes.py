import cv2
print(cv2.__version__)


def frame_capture(file):
    vidcap = cv2.VideoCapture(file)
    success,image = vidcap.read()
    print(vidcap)
    #print(image.shape)
    global count
    success = True
    while success:
        cv2.imwrite("hit%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print 'Read a new frame: ', success
        count +=1
        print(count)

import os
count = 0
for file in os.listdir("/home/malar/Desktop/hit/hit"):
    if file.endswith(".avi"):
        path=os.path.join("/home/malar/Desktop/hit/hit", file)
        frame_capture(path)

  
  
