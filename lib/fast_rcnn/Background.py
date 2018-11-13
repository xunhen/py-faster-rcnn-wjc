import cv2
import numpy as np
import time
class Background:
    def __init__(self,img_name='',threshold=0.2):
        self.threshold=threshold
        self.img_name=img_name

    def read(self):
        self.im=cv2.imread(self.img_name,cv2.IMREAD_GRAYSCALE)
        _,self.im = cv2.threshold(self.im, 127, 255, 0)
        return self.im

    def resize(self,im_sacle):
        self.im = cv2.resize(self.im, None, None, fx=im_sacle, fy=im_sacle,
                        interpolation=cv2.INTER_LINEAR)

    def processAndgenerate(self,im_sacle=1):
        self.read()
        self.resize(im_sacle)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        self.im=cv2.morphologyEx(self.im, cv2.MORPH_CLOSE, kernel)
        self.im = cv2.morphologyEx(self.im, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(self.im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.boxes = []
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            self.boxes.append([x, y, x + w, y + h])
        self.boxes=np.array(self.boxes)

    def filter(self,proposals=np.array([])):
        org_len=len(proposals)
        start=time.time()
        areas=np.zeros([len(proposals)],dtype=float)
        for i in range(len(proposals)):
            for box in self.boxes:
                areas[i]+=self.intern_area(proposals[i],box)
            areas[i]/=self._area(proposals[i])
        keep=np.where(areas>=self.threshold)[0]
        end=time.time()
        print(org_len,len(keep),org_len-len(keep),(end-start)*1000)
        return np.where(areas>=self.threshold)[0]


    def intern_area(self,proposal,box):
        x1=np.maximum(proposal[0],box[0])
        y1=np.maximum(proposal[1],box[1])
        x2 = np.minimum(proposal[2], box[2])
        y2 = np.minimum(proposal[3], box[3])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        return w*h

    def _area(self,box):
        w = np.maximum(0.0, box[2] - box[0] + 1)
        h = np.maximum(0.0, box[3] - box[1] + 1)
        return w*h
if __name__ == '__main__':
    backgound=Background('F:/PostGraduate/Projects/background/video/pre/2.jpg',0.5)
    backgound.processAndgenerate(0.5)
    cv2.waitKey()
