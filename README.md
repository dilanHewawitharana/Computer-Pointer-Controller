# Computer Pointer Controller

In this project, I used a gaze detection model to control the mouse pointer of the computer. I used the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate the ability to run multiple models in the same machine and coordinate the flow of data between those models.

I used the InferenceEngine API from Intel's OpenVino ToolKit to build the project. The [gaze estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) requires three inputs:

- The head pose
- The left eye image
- The right eye image.

To get these inputs, you will have to use three other OpenVino models:

* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

### The Pipeline

I had to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data will look like this:

![architectural diagram](./images/pipeline.png)

## Project Set Up and Installation

| Details            |              |
|-----------------------|---------------|
| Programming Language |  Python 3.6.5 |
| OpenVino Version |  openvino_2020.1.033 |
| Environment |  Windows 10 |
| Hardware used |  CORE i7 CPU - 7th Gen |

- First, You need to install the prerequisites from requirements.txt using the following command.

```
pip install -r requirements.txt
```
### Download the pre-trained models

- Go to the Model Downloader directory: 

```
cd <OpenVINO-Path>\IntelSWTools\openvino\deployment_tools\tools\model_downloader
```

**1. Download Face Detection Model**

```
python downloader.py --name face-detection-adas-binary-0001
```

**2. Download Facial Landmarks Detection Model**

```
python downloader.py --name landmarks-regression-retail-0009
```

**3. Download Head Pose Estimation Model**

```
python downloader.py --name head-pose-estimation-adas-0001
```

**4. Download Gaze Estimation Model**

```
python downloader.py --name gaze-estimation-adas-0002
```

- Copy the downloaded models into project folder so that the folder structure as following root tree.(you can remove unused models after you finally select most suitable model for your project)

.
├───.Instructions.md.swp
├───Log.log
├───README.md
├───requirements.txt
│
├───bin
│   ├───.gitkeep
│   ├───demo.mp4
│
├───images
│   ├───pipeline.png
│
├───models
│   └───intel
│       ├───face-detection-adas-binary-0001
│       │   └───FP32-INT1
│       │           ├───face-detection-adas-binary-0001.bin
│       │           ├───face-detection-adas-binary-0001.xml
│       │
│       ├───gaze-estimation-adas-0002
│       │   ├───FP16
│       │   │       ├───gaze-estimation-adas-0002.bin
│       │   │       ├───gaze-estimation-adas-0002.xml
│       │   │
│       │   ├───FP16-INT8
│       │   │       ├───gaze-estimation-adas-0002.bin
│       │   │       ├───gaze-estimation-adas-0002.xml
│       │   │
│       │   └───FP32
│       │           ├───gaze-estimation-adas-0002.bin
│       │           ├───gaze-estimation-adas-0002.xml
│       │
│       ├───head-pose-estimation-adas-0001
│       │   ├───FP16
│       │   │       ├───head-pose-estimation-adas-0001.bin
│       │   │       ├───head-pose-estimation-adas-0001.xml
│       │   │
│       │   ├───FP16-INT8
│       │   │       ├───head-pose-estimation-adas-0001.bin
│       │   │       ├───head-pose-estimation-adas-0001.xml
│       │   │
│       │   └───FP32
│       │           ├───head-pose-estimation-adas-0001.bin
│       │           ├───head-pose-estimation-adas-0001.xml
│       │
│       └───landmarks-regression-retail-0009
│           ├───FP16
│           │       ├───landmarks-regression-retail-0009.bin
│           │       ├───landmarks-regression-retail-0009.xml
│           │
│           ├───FP16-INT8
│           │       ├───landmarks-regression-retail-0009.bin
│           │       ├───landmarks-regression-retail-0009.xml
│           │
│           └───FP32
│                   ├───landmarks-regression-retail-0009.bin
│                   ├───landmarks-regression-retail-0009.xml
│
└───src
    ├───face_detection.py
    ├───facial_landmarks_detection.py
    ├───gaze_estimation.py
    ├───head_pose_estimation.py
    ├───inference.py
    ├───input_feeder.py
    ├───main.py
    ├───mouse_controller.py
    │
    └───__pycache__
        ├───face_detection.cpython-36.pyc
        ├───facial_landmarks_detection.cpython-36.pyc
        ├───gaze_estimation.cpython-36.pyc
        ├───head_pose_estimation.cpython-36.pyc
        ├───inference.cpython-36.pyc
        ├───input_feeder.cpython-36.pyc
        ├───mouse_controller.cpython-36.pyc


## Demo

1. Initialize the OpenVINO environment 

```
cd <OpenVINO-Path>\IntelSWTools\openvino\bin\
setupvars.bat
```

2. cd into project folder

```
cd <Project-Repo-Path>
```
3. Run the main.py 

- For sample video

```
python src\main.py -fdm models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml 
-flm models\intel\landmarks-regression-retail-0009\FP16-Int8\landmarks-regression-retail-0009.xml 
-hpm models\intel\head-pose-estimation-adas-0001\FP16-Int8\head-pose-estimation-adas-0001.xml 
-gem models\intel\gaze-estimation-adas-0002\FP16-Int8\gaze-estimation-adas-0002.xml 
-i bin\demo.mp4 -d CPU
```

- For camera mode

```
python src\main.py -fdm models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml 
-flm models\intel\landmarks-regression-retail-0009\FP16-Int8\landmarks-regression-retail-0009.xml 
-hpm models\intel\head-pose-estimation-adas-0001\FP16-Int8\head-pose-estimation-adas-0001.xml 
-gem models\intel\gaze-estimation-adas-0002\FP16-Int8\gaze-estimation-adas-0002.xml 
-i CAM -d CPU
```

## Documentation

- Command Line Arguments

| Argument            | Description     | Type          |
|---------------------|-----------------|---------------|
| -fdm | Path to an .xml file with Face Detection model. | required |
| -flm | Path to an .xml file with Facial Landmark Detection model. | required |
| -hpm | Path to an .xml file with Head Pose Estimation model. | required |
| -gem | Path to an .xml file with Gaze Estimation model. | required |
| -i | Path to image or video file or CAM | optional |
| -l | cpu_extension | optional |
| -d | Specify the target device to infer on | required |

## Benchmarks

This program tested on Windows 10 environment and  Intel CORE i7 - 7th Gen CPU use as hardware .

### Loading time for each model (device is CPU)

| Model | Face Detection | Facial Landmark Detection | Head Pose Estimation | Gaze Estimation |
|---------------------|-----------------|---------------|---------------|---------------|
|FP16-INT8 | NA | 136.7ms | 706.1ms | 727.1ms |
|FP16 | NA |  478.7ms | 210.3ms | 519.3ms |
|FP32 | 285.2ms | 467.9ms | 204.1ms | 545.6ms |


### Infernece time for each model (device is CPU)

| Model | Face Detection | Facial Landmark Detection | Head Pose Estimation | Gaze Estimation |
|---------------------|-----------------|---------------|---------------|---------------|
|FP16-INT8 | NA | 0.6ms | 1.3ms | 1.7ms |
|FP16 | NA | 0.7ms | 1.5ms | 1.9ms |
|FP32 | 16.8ms | 0.8ms | 1.5ms | 2.1ms |

## Results

When considering the inference time, FP32 has the highest inference time and INT8 has lowest inference time. FP16 has a middle value between FP32 and FP16.

## Stand Out Suggestions

- There are 4 models used in this program. These models have different precision levels. This project is tested with all models with different precision levels and it is supported for every model with different precision levels. 

- Program gives high accuracy when using the FP32 model. It gives lowest accuracy when using the INT8 model.

- Program gives high speed when using INT8 model, while gives lowest speed when using FP32 model. 

### Edge Cases

There is a situation where this program breaks When the mouse pointer goes out from the screen, Then pyautogui.FAILSAFE value has changed into false even though it is not recommended.

Lighting may cause a change in the accuracy of the program but it does not cause the program to break.

