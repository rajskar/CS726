import cv2
print(cv2.__version__)

import matplotlib.pyplot as plt


video_capture = cv2.VideoCapture('/media/rajs/Elements/NG canteen/Export Folder(18)/CniD(1).avi')       

while cv2.waitKey(10) < 0:    

    s1 = []
    f = []
    for j in range(5):
        _, frame = video_capture.read()
        frame = cv2.resize(frame, (320, 240))       
        f.append(frame)
        if j !=0:
            cv2.imshow('Contours' +str(j), f[j] - f[j-1])
        
        cv2.waitKey(10)
        
    cv2.waitKey(0)
            
cv2.destroyAllWindows()
print('Released')
plt.close()
