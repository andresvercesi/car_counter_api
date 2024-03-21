#from email.mime import image
#from pickletools import float8
from fastapi import FastAPI, Query, UploadFile, BackgroundTasks 
from fastapi.responses import FileResponse
import cv2
import torch
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from vidgear.gears import CamGear
import requests
#import typing
from typing import Annotated
from functions import remove_file, remove_folder, results_count, results_show

#modelYOLO8n = YOLO('models\yolov8n.pt')  # Load pretrained YOLOv8n model
#modelYOLO8s = YOLO('models\yolov8s.pt')  # Load pretrained YOLOv8s model
#modelYOLO8m = YOLO('models\yolov8m.pt')  # Load pretrained YOLOv8m model
model = YOLO('models\yolov9c.pt')  # Load pretrained YOLOv9c model


app = FastAPI()

@app.get("/")
async def readme():
    '''
    Return a API name and description
    '''
    return ("Object count with YOLO Model API - Go to /docs to see API documentation")

@app.get("/class_names")
async def class_names():
    '''
    Returns a list of classes detected by model
    '''
    return (model.names)


@app.post("/objects_detection_count_local_image")
async def count_objects_file(file: UploadFile, 
                            class_filter : Annotated[list[int] | None, Query()] = None,
                            conf : float = 0.25):
    '''
    Return a list of objects detected in a image file with quantities
    '''
    with open(file.filename, 'wb') as disk_file:
        file_bytes = await file.read()
        disk_file.write(file_bytes)
    results = model.predict(disk_file.name, conf=conf, classes=class_filter)  # return a list of Results objects
    remove_file(disk_file.name) #Remove a upload file after detection
    return results_count(results)

@app.get("/objects_detection_count_url_image")
async def count_objects_url(url: str, 
                            class_filter : Annotated[list[int] | None, Query()] = None,
                            conf : float = 0.25):
    '''
    Return a list of objects detected in a url image with quantities
    '''
    response = requests.get(url)
    with open("image.jpg", "wb") as disk_file:
        disk_file.write(response.content)
    # Run batched inference on a list of images
    results = model.predict(disk_file.name, conf=conf, classes=class_filter)  # return a list of Results objects
    os.remove(results[0].path)
    return results_count(results)

@app.post("/objects_detection_show_local_image")
async def show_objects_file(file: UploadFile,
                            class_filter : Annotated[list[int] | None, Query()] = None,
                            conf : float = 0.25):
    '''
    Return a image with bounding box around objects detected in a image file
    '''
    path_result=''
    file_name=''
    with open(file.filename, 'wb') as disk_file:
        file_bytes = await file.read()
        disk_file.write(file_bytes)
    results = model.predict(disk_file.name, conf=conf, save=True, classes= class_filter)  # return a list of Results objects
    file_name = (os.path.basename(results[0].path))
    run_folder = results[0].save_dir
    path_result = run_folder+"/"+file_name
    remove_file(file_name)
    tasks = BackgroundTasks()
    tasks.add_task(remove_file, path=path_result)
    tasks.add_task(remove_folder, path=run_folder)
    return FileResponse(path_result, background=tasks)
 

@app.get("/objects_detection_show_url_image")
async def show_objects_url(url: str, 
                            class_filter : Annotated[list[int] | None, Query()] = None,
                            conf : float = 0.25):
    '''
    Return a image with bounding box around objects detected in a url image
    '''
    results = 0
    response = requests.get(url)
    with open("image.jpg", "wb") as disk_file:
        disk_file.write(response.content)
    results = model.predict(disk_file.name, conf=conf, save=True, classes=class_filter)  # return a list of Results objects
    file_name = (os.path.basename(results[0].path))
    run_folder = results[0].save_dir
    path_result = run_folder+"/"+file_name
    remove_file(file_name)
    tasks = BackgroundTasks()
    tasks.add_task(remove_file, path=path_result)
    tasks.add_task(remove_folder, run_folder)
    return FileResponse(path_result, background=tasks)

"""
@app.get("/objects_count_stream")
async def objects_detect(path: str, class_filter : int = None):
    video_path = path
    cap = cv2.VideoCapture(path)
    # Loop through the video frames
    i = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        i=i+1
        if success:
            # Run YOLOv8 inference on the frame
            if i%1000==0:
                results = model(frame)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return

@app.get("/objects_count_local_video")
async def objects_detect(path: str, 
                        class_filter : Annotated[list[int] | None, Query()] = None,
                        conf : float = 0.25):
    frame_set_no = 150
    results = 0
    video_path = path
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_set_no)
    success, frame = cap.read()
    results = model.predict(frame, conf=conf, save=True, classes= class_filter)  # return a list of Results objects
    file_name = (os.path.basename(results[0].path))
    path_result = results[0].save_dir+"/"+file_name
    tasks = BackgroundTasks()
    tasks.add_task(remove_file, path=path_result)
    return FileResponse(path_result, background=tasks)

@app.get("/objects_count_webcam")
async def objects_detect(path: str, 
                        class_filter : Annotated[list[int] | None, Query()] = None,
                        conf : float = 0.25):
    frame_set_no = 50
    results = 0
    video_path = path
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_set_no)
    success, frame = cap.read()
    results = model.predict(frame, conf=conf, save=True, classes= class_filter)  # return a list of Results objects
    file_name = (os.path.basename(results[0].path))
    path_result = results[0].save_dir+"/"+file_name
    tasks = BackgroundTasks()
    tasks.add_task(remove_file, path=path_result)
    return FileResponse(results, background=tasks)
   


@app.get("/objects_count_youtube")
async def objects_detect(url: str):
    url = 'https://'+url
    stream = CamGear(source=url, stream_mode = True, logging=True, **options_stream).start() # YouTube Video URL as input
    i=0
    while True:
        frame = stream.read()
        i=i+1
        # read frames
        # check if frame is None
        if frame is None:
            #if True break the infinite loop
            break
        
        if i%100==0:
            results = modelYOLO(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

        # do something with frame here
    
        #cv2.imshow("Output Frame", frame)
        # Show output window

        key = cv2.waitKey(1) & 0xFF
        # check for 'q' key-press
        if key == ord("q"):
            #if 'q' key-pressed break out
            break

    cv2.destroyAllWindows()
    # close output window

    # safely close video stream.
    stream.stop()
    return """