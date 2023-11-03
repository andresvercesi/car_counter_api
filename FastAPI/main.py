from email.mime import image
from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

modelYOLO = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model

def object_detection(image):
    pred = model(image)
    #xmin,ymin,xmax,ymax
    df=pred.pandas().xyxy[0]
    #Filter by confidence
    df=df[df["confidence"]>0.01]
    df2=df.groupby(['name'])['name'].count()
    json_data = df2.to_dict()
    return json_data



app = FastAPI()

""" @app.get("/objects_count_local")
async def objects_detect(path: str):
    image = cv2.imread(path)
    pred = model(image)
    #xmin,ymin,xmax,ymax
    df=pred.pandas().xyxy[0]
    #Filter by confidence
    df=df[df["confidence"]>0.01]
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
    json_data = df2.to_dict()
    #json_data = df[['class', 'name']].to_json(orient='records')
    return json_data """

@app.post("/object_count_load/")
async def load_image(imagefile: UploadFile = File(...)):
    
    folder = 'load_images'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    complete_path = os.path.join(folder, imagefile.filename)

    with open(complete_path, "wb") as f:
        f.write(imagefile.file.read())

    image = cv2.imread(complete_path)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = object_detection(image)
    print(result)

    return {
        "nombre": imagefile.filename,
        "tipo": imagefile.content_type,
        "tamaño": len(imagefile.file.read()),
    }

""" @app.get("/objects_count_local_fast")
async def objects_detect(path: str):
    image = cv2.imread(path)
    pred = model(image)
    #xmin,ymin,xmax,ymax
    df=pred.pandas().xyxy[0]
    #Filter by confidence
    df=df[df["confidence"]>0.01]
    df2=df.groupby(['name'])['name'].count()
    json_data = df2.to_dict()
    #json_data = df[['class', 'name']].to_json(orient='records')
    return json_data """

@app.get("/objects_count_local_fast")
async def objects_detect(path: str):
    image = cv2.imread(path)
    # Run batched inference on a list of images
    results = modelYOLO.predict(path, classes=2, conf=0.25)  # return a list of Results objects
    return results[0].tojson()

@app.get("/objects_count_local")
async def objects_detect(path: str):
    image = cv2.imread(path)
    # Run batched inference on a list of images
    results = modelYOLO.predict(path, classes=2, conf=0.25)  # return a list of Results objects
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        print(r.probs)
    return r.tojson()