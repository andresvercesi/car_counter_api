import os

def results_count(inference_results):
    """Return a dictionary with classes and quantities detected
    
    Args:
        inference_results (YOLO predict out)

    Returns:
        dict
    """
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

def results_show(inference_results) -> None:
    """Show image with bounding box for objects detected
    
    Args:
        inference_results (YOLO predict out)

    Returns:
    """
    for result in inference_results:
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image

def remove_file(path:str) -> None:
    """Remove a file
    
    Args:
        path: str

    Returns:
    """
    try:
        os.remove(path)
    except OSError as e:
        print(f'Error trying delete file: {e}')

def remove_folder(path:str) -> None:
    """Remove a folder
    
    Args:
        path: str

    Returns:
    """
    try:
        os.rmdir(path)
    except OSError as e:
        print(f'Error trying delete file: {e}')
