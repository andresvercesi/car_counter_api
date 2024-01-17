# Objects counter API
### Introduction
Objects counter API is an API to detect objects in a image using YOLO pre trained model and FastAPI
### Objects counter API Features
* Users can count all detected objects, or only selected classes
### Installation Guide
* Clone this repository [here](https://github.com/andresvercesi/objects_counter_api).
* The main branch is the most stable branch at any given time, ensure you're working from it.
* Could you test the demo deployed in https://objects-counter.onrender.com
### Usage
* 
* 
### API Endpoints
| HTTP Verbs | Endpoints | Action |

| GET | /api/docs | Access to API documentation |

| GET | /api/objects_detection_show_url_image?url=(URL) | Detect all objects in a URL image and show in the web browser |

| GET | /api/objects_detection_count_url_image?url=(URL) | Detect all objects in a URL image and return results |

### Technologies Used
* [FastAPI](https://fastapi.tiangolo.com/) FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

* [YOLO](https://docs.ultralytics.com/) This is a real-time object detection and image segmentation model.

### Authors

Andres Vercesi

### License

