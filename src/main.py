import cv2
import math
import time

from argparse import ArgumentParser
import logging as log

from input_feeder import InputFeeder
from win32api import GetSystemMetrics

from face_detection import Model_Face_Detection
from facial_landmarks_detection import Model_Facial_Landmarks_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from mouse_controller import MouseController


def build_argparser():
    '''
    Parse command line arguments.

    :return: command line arguments
    '''
    parser = ArgumentParser()
    
    parser.add_argument("-fdm", "--face_detection_model", required=True, type=str,
                        help="Path to an .xml file with Face Detection model.")
    parser.add_argument("-flm", "--facial_landmark_model", required=True, type=str,
                        help="Path to an .xml file with Facial Landmark Detection model.")
    parser.add_argument("-hpm", "--head_pose_model", required=True, type=str,
                        help="Path to an .xml file with Head Pose Estimation model.")
    parser.add_argument("-gem", "--gaze_estimation_model", required=True, type=str,
                        help="Path to an .xml file with Gaze Estimation model.")
                        
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
                        
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
                        
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    
    return parser


def init_log():
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            log.FileHandler("Log.log"),
            log.StreamHandler()
    ])

def main():
    #Initialize log
    init_log()

    # Grab command line args
    args = build_argparser().parse_args()

    # Model initialize
    fd_model = Model_Face_Detection(args.face_detection_model, args.device, args.cpu_extension)
    fl_model = Model_Facial_Landmarks_Detection(args.facial_landmark_model, args.device, args.cpu_extension)
    ge_model = Model_Gaze_Estimation(args.gaze_estimation_model, args.device, args.cpu_extension)
    hp_model = Model_Head_Pose_Estimation(args.head_pose_model, args.device, args.cpu_extension)

    # Model load
    log.info("===Models Load time ===") 
    start_time = time.time()
    fd_model.load_model()
    log.info("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    fl_model.load_model()
    log.info("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    ge_model.load_model()
    log.info("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    hp_model.load_model()
    log.info("Head Pose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    log.info("All Model loaded...")

    mouse = MouseController("low","fast")
    
    ### Handle the input stream ###
    feed = None
    # Camera input stream
    if args.input == 'CAM':
        feed = InputFeeder("cam")
    # Image input
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        feed = InputFeeder("image", args.input)
    # Video input stream
    else:
        feed = InputFeeder("video", args.input)

    feed.load_data()
    log.info("Input stream initialized...")

    frame_count = 0
    fd_inference_time = 0
    hp_inference_time = 0
    fl_inference_time = 0
    ge_inference_time = 0

    # Window initialized
    cv2.namedWindow('video')        
    cv2.moveWindow('video', int(math.floor(GetSystemMetrics(0)/2)-360),int(math.floor(GetSystemMetrics(1)/2)-240))  

    for frame in feed.next_batch():
        if frame is None:
            break

        # detect face from frame
        p_frame = fd_model.preprocess_input(frame)
        start_time = time.time()
        output = fd_model.predict(p_frame)
        fd_inference_time += time.time() - start_time
        frame, detected_faces = fd_model.preprocess_output(output, frame)

        for face in detected_faces:
            # crop face from frame
            cropped_face = frame[face[1]:face[3],face[0]:face[2]]

            # detect head pose
            p_frame = hp_model.preprocess_input(cropped_face)
            start_time = time.time()
            output = hp_model.predict(p_frame)
            hp_inference_time += time.time() - start_time
            frame, headpose_angel = hp_model.preprocess_output(output, frame)
            
            # detect facial landmarks
            p_frame = fl_model.preprocess_input(cropped_face)
            start_time = time.time()
            output = fl_model.predict(p_frame)
            fl_inference_time += time.time() - start_time
            frame, left_eye_point, right_eye_point = fl_model.preprocess_output(output, face, frame)

            cropped_left_eye = fl_model.crop_eye(left_eye_point, cropped_face)
            cropped_right_eye = fl_model.crop_eye(right_eye_point, cropped_face)
       
            # detect gaze estimation
            p_left_eye = ge_model.preprocess_input(cropped_left_eye)
            p_right_eye = ge_model.preprocess_input(cropped_right_eye)
            start_time = time.time()
            output = ge_model.predict(headpose_angel, p_left_eye, p_right_eye)
            ge_inference_time += time.time() - start_time
            gazevector = ge_model.preprocess_output(output)

        frame_count += 1
        if(frame_count % 5 == 0):
            # move mouse pointer
            mouse.move(gazevector[0],gazevector[1])

        cv2.imshow('video',cv2.resize(frame,(720,480)))

        key = cv2.waitKey(60)
        if key==27:
            break

    #logging inference times
    if(frame_count>0):
        log.info("=== Model Inference time ===") 
        log.info("Face Detection:{:.1f}ms".format(1000* fd_inference_time/frame_count))
        log.info("Facial Landmarks Detection:{:.1f}ms".format(1000* fl_inference_time/frame_count))
        log.info("Headpose Estimation:{:.1f}ms".format(1000* hp_inference_time/frame_count))
        log.info("Gaze Estimation:{:.1f}ms".format(1000* ge_inference_time/frame_count))

    log.info("VideoStream ended...")
    cv2.destroyAllWindows()
    feed.close()

if __name__ == '__main__':
    main() 
 