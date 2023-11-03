import cv2
import torch
import numpy as np
from PIL import Image
#import matplotlib.path as mplPath

##model= torch.load('yolov5s.pt')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
image_path = 'C:\Dev\CarCounter\Data\images (1).jpg'


def car_count(image_path):
    image = cv2.imread(image_path)
    
    pred = model(image)
    #xmin,ymin,xmax,ymax
    df=pred.pandas().xyxy[0]
    #Filter by confidence
    df=df[df["confidence"]>0.1]
    df2=df.groupby(['name'])['name'].count()
    
    for i in range(df.shape[0]):
        bbox = df.iloc[i][["xmin","ymin","xmax","ymax"]].values.astype(int)
        #Draw Bounding Box
        cv2.rectangle(image, (bbox[0], bbox[1]),(bbox[2],bbox[3]), (255,0,0), 2)
        #Print object class
        cv2.putText(image,
                    f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'],4)}",
                    (bbox[0], bbox[1]-15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255,255,255),
                    2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(df2['car'])

print(car_count(image_path))
