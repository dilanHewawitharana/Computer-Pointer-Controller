'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from inference import Network
import cv2

class Model_Face_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        set your instance variables.
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

    def preprocess_output(self, outputs, frame, prob_threshold=0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        height = frame.shape[0]
        width = frame.shape[1]
        detected_faces = []
        probs = outputs[0, 0, :, 2]
        for i, p in enumerate(probs):
            if p > prob_threshold:
                box = outputs[0, 0, i, 3:]
                p1_x = int(box[0] * width)
                p1_y = int(box[1] * height)
                p2_x = int(box[2] * width)
                p2_y = int(box[3] * height)
                detected_faces.append([p1_x, p1_y, p2_x, p2_y])
                frame = cv2.rectangle(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
        return frame, detected_faces
