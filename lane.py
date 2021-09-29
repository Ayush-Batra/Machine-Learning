import cv2 as cv
import numpy as np

cap = cv.VideoCapture('igvc.mp4')
class Lane_Detection:
    def __init__(self,frame):
        self.frame = frame
    def copy_image(self,frame):
        copied = frame.copy()
        print("copy")
        return copied
    def change(self,frame):
        frame[:,:,1] = 0
        frame[:,:,2] = 0
        #cv.imshow("blue_frame",frame)
        print("change")
        return frame
    def blurring(self,frame):
        kernel = (5,5)
        blurred = cv.GaussianBlur(frame,kernel,0)
        cv.imshow("blue_blurred_frame",blurred)
        print("blur")
        return blurred
    def threshold(self,frame):
        ret,thresh = cv.threshold(frame,120,255,cv.THRESH_BINARY)
        print("thresh")
        return thresh
    def dilation(self,frame):
        kernel = (3,3)
        #dilate = cv.dilate(frame,kernel,iterations = 1)
        opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
        print("dilate")
        return opening
    def edge_detection(self,frame):
        edge = cv.Canny(frame,0,200)
        print("edge")
        return edge
    def ROI(self,frame,vertices) :
        mask=np.zeros_like(frame)  
        match_mask_color=(255,)
        cv.fillPoly(mask,vertices,match_mask_color)
        masked=cv.bitwise_and(frame,mask)
        print("roi")
        return masked
    def hough_lines(self,roi, frame):
        lines = cv.HoughLinesP(roi,1,np.pi/180,15,minLineLength=20,maxLineGap=15)
        for line in lines:
            for x1,y1,x2,y2 in line:
                frames = cv.line(frame,(x1,y1),(x2,y2),(255,0,0),3)
        lines_edges = cv.addWeighted(frames, 0.8, frame, 1, 0)
        print("hough")
        return lines_edges
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    lane = Lane_Detection(frame)
    height=frame.shape[0]
    width=frame.shape[1]
    colored = lane.copy_image(frame)
    colored = lane.change(colored)
    blur = lane.blurring(colored)
    thresh = lane.threshold(blur)
    dilate = lane.dilation(thresh)
    edges = lane.edge_detection(dilate)
    ROI_vertices=[(0,height),(width,height),(width,height-300),(0,height-300)]
    ROI_image=lane.ROI(edges,np.array([ROI_vertices],np.int32))
    lines_edges = lane.hough_lines(ROI_image, frame)
    #cv.imshow('frame', frame)
    #cv.imshow('dilate', dilate)
    #cv.imshow('edged', edges) 
    cv.imshow("Lane",lines_edges)
    if cv.waitKey(3) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
cap.isOpened()
