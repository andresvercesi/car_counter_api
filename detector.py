import cv2
import torch
import numpy as np
from PIL import Image
#import matplotlib.path as mplPath


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def detector():
    image = cv2.imread("C:\Dev\CarCounter\Data\car_on_street1.jpg")
    cap = cv2.VideoCapture("C:\Dev\CarCounter\Data\people.mp4")
    
    while cap.isOpened():
        status , frame = cap.read()

        if not status:
            break

        pred = model(frame)

        #xmin,ymin,xmax,ymax
        df=pred.pandas().xyxy[0]
        #Filter by confidence
        df=df[df["confidence"]>0.6]

        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin","ymin","xmax","ymax"]].values.astype(int)
            #Draw Bounding Box
            cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[2],bbox[3]), (255,0,0), 2)
            #Print object class
            cv2.putText(frame,
                        f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'],4)}",
                        (bbox[0], bbox[1]-15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255,255,255),
                        2)

        cv2.imshow("video", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()

    

detector()
