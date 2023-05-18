#OpenCV
import cv2 as cv
import numpy as np
capture = cv.VideoCapture(0)

#Timer 
import time 
starttime =time.time()
totaltime= 0
#Counting people
inCar = 0
#motion exists
motion = 0
lx = 0
ly = 0

#void
count =0

def rescaleFrame(frame,scale=.75):
    width=int(frame.shape[1] *scale)
    height=int(frame.shape[0]*scale)
    dimensions =(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


#Video Feed 
while True:
    
    ret,frame  =capture.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)
    for(x,y,w,h) in faces_rect:
      cv.rectangle(gray,(x,y),(w+x,y+h),(255,255,255),thickness=2)
      #counts the amount of faces it detects
      motion = len(faces_rect)
      #print("")
      cv.imshow('frame',gray)
      if(count==0):
        lx=x
        ly=y
        count+=1
         
      #Writing frame data to countCheckers
      if(lx<x+5 & ly<y+5):
         inCar-=1
         count=0
         if(inCar<0):
            inCar=0
         print("There is a decrease. New total: "+ str(inCar))
     
     #IF we go up there is a person gone, so closer to camera = gone
      if(lx>x+10 & ly>y+10):
         inCar+=1
         count=0
         print("There is a increase. New total: "+ str(inCar))
      else:
         lx=x
         ly=y
      #Checks if anyone bounding boxes exist
      if(x,y,w,h == None):
         motion=0
      
      if(inCar>=-1):
         with open('inTrain.txt','a') as g:
            g.write(str(int(inCar)))
            g.write('\n')
            g.close()
      #Timer Updates 
      totaltime =round((time.time()-starttime),0)
      starttime = time.time()

    #Writing to our file when no more motion exists 
    if(motion == 0):
         with open('wait_time.txt', 'a') as f:
          if(totaltime!=0):
             f.write(str(int(totaltime)))
             f.write('\n')   
             f.close()
    #Close cams
    if(cv.waitKey(1) & 0xFF ==ord('d')):
        break
capture.release()
cv.destroyAllWindows()
cv.waitKey(0)