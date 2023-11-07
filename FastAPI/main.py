from email.mime import image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import cv2
import torch
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from vidgear.gears import CamGear

def results_count(inference_results):
    results_dict = {}
    for result in inference_results:
        detection_count = result.boxes.shape[0]        
        for i in range(detection_count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            if name in results_dict:
                results_dict[name] +=1
            else:
                results_dict[name] =1
            #confidence = float(result.boxes.conf[i].item())
            #bounding_box = result.boxes.xyxy[i].cpu().numpy()

            #x = int(bounding_box[0])
            #y = int(bounding_box[1])
            #width = int(bounding_box[2] - x)
            #height = int(bounding_box[3] - y)
    return results_dict 

def results_show(inference_results):
    for result in inference_results:
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image

def remove_file(path:str) -> None:
    os.remove(path)

modelYOLO = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model

app = FastAPI()

@app.post("/objects_detection_count_local_image")
async def post_media_file(file: UploadFile):
    with open(file.filename, 'wb') as disk_file:
        file_bytes = await file.read()
        disk_file.write(file_bytes)
    results = modelYOLO.predict(disk_file.name, conf=0.25)  # return a list of Results objects
    remove_file(disk_file.name)
    return results_count(results)

@app.get("/objects_detection_count_url_image")
async def objects_detect(url: str):
    url = 'https://'+url
    # Run batched inference on a list of images
    results = modelYOLO.predict(url, conf=0.25)  # return a list of Results objects
    os.remove(results[0].path)
    return results_count(results)

@app.post("/objects_detection_show_local_image")
async def post_media_file(file: UploadFile):
    path_result=''
    file_name=''
    with open(file.filename, 'wb') as disk_file:
        file_bytes = await file.read()
        disk_file.write(file_bytes)
    results = modelYOLO.predict(disk_file.name, conf=0.25, save=True)  # return a list of Results objects
    remove_file(disk_file.name)
    file_name = (os.path.basename(results[0].path))
    path_result = results[0].save_dir+"/"+file_name
    tasks = BackgroundTasks()
    tasks.add_task(remove_file, path=path_result)
    return FileResponse(path_result, background=tasks)


@app.get("/objects_detection_show_url_image")
async def objects_detect(url: str):
    results = 0
    url = 'https://'+url
    # Run batched inference on a list of images
    results = modelYOLO.predict(url, conf=0.25, save=True)  # return a list of Results objects
    #results_show(results)
    file_name = (os.path.basename(results[0].path))
    path_result = results[0].save_dir+"/"+file_name
    print(path_result)
    remove_file(file_name)
    tasks = BackgroundTasks()
    tasks.add_task(remove_file, path=path_result)
    return FileResponse(path_result, background=tasks)

@app.get("/objects_count_stream")
async def objects_detect(path: str):
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
                results = modelYOLO(frame)

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

""" @app.get("/objects_count_youtube")
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