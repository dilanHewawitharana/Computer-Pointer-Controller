'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from inference import Network
import cv2

class Model_Facial_Landmarks_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initialize instance variables.
        '''
        self.model = model_name
        self.device =  device
        self.extensions = extensions
   
        self.infer_network = Network()

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.infer_network.load_model(self.model, self.extensions, self.device)

    def predict(self, image, request_id=0):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.infer_network.exec_net(image, request_id)

        if self.infer_network.wait() == 0:
            ### Get the results of the inference request ###
            result = self.infer_network.get_output(output_type='output_blob')
            return result

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.infer_network.get_input_shape()

        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame

    def preprocess_output(self, outputs, face, frame):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # for landmarks-regression_retail-0009
        normed_landmarks = outputs.reshape(1, 10)[0]

        height = face[3]-face[1]
        width = face[2]-face[0]
        
        for i in range(2):
            x = int(normed_landmarks[i*2] * width)
            y = int(normed_landmarks[i*2+1] * height)
            # Drawing the circle in the image
            cv2.circle(frame, (face[0]+x, face[1]+y), 30, (0,255,255), 2)
        
        left_eye_point =[normed_landmarks[0] * width, normed_landmarks[1] * height]
        right_eye_point = [normed_landmarks[2] * width, normed_landmarks[3] * height]
        return frame, left_eye_point, right_eye_point

    def crop_eye(self, eye_point, cropped_face):
        Xmin = int(eye_point[0] - 30) if  int(eye_point[0] - 30) >=0 else 0 
        Xmax = int(eye_point[0] + 30) if  int(eye_point[0] + 30) >=0 else 0 
        ymin = int(eye_point[1] - 30) if  int(eye_point[1] - 30) >=0 else 0 
        ymax = int(eye_point[1] + 30) if  int(eye_point[1] + 30) >=0 else 0 

        #print(str(Xmin) + ' : ' + str(Xmax) +' : ' + str(ymin)  +' : ' + str(ymax))
        cropped_eye = cropped_face[ymin:ymax,Xmin:Xmax]
        return cropped_eye
