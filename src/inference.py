#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, CPU_EXTENSION, DEVICE, console_output= False):
        ### Load the model ###
        ### Check for supported layers ###
        ### Add any necessary extensions ###
        ### Return the loaded inference plugin ###
        
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### Check for supported layers ###
        if not all_layers_supported(self.plugin, self.network, console_output=console_output):
            ### Add any necessary extensions ###
            self.plugin.add_extension(CPU_EXTENSION, DEVICE)
            
        self.exec_network = self.plugin.load_network(self.network, DEVICE)
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        ### Return the loaded inference plugin ###
        return


    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image, request_id):
        ### Start an asynchronous request ###
        ### Return any necessary information ###
        self.exec_network.start_async(request_id, 
            inputs={self.input_blob: image})
        return

    def exec_net_for_gaze_estimation(self, headpose_angel, left_eye, right_eye, request_id):
        ### Start an asynchronous request ###
        ### Return any necessary information ###
        inputiter = iter(self.network.inputs)
        next(inputiter)#ignore first input
        self.exec_network.start_async(request_id, 
            inputs={self.input_blob: headpose_angel, next(inputiter):left_eye, next(inputiter):right_eye})
        return

    def wait(self):
        ### Wait for the request to be complete. ###
        ### Return any necessary information ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self, output_type=None):
        ### Extract and return the output results
        if output_type is 'output_blob':
            return self.exec_network.requests[0].outputs[self.output_blob]
        else:
            return self.exec_network.requests[0].outputs
    
def all_layers_supported(engine, network, console_output=False):
    ### check if all layers are supported
    ### return True if all supported, False otherwise
    layers_supported = engine.query_network(network, device_name='CPU')
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False

    return all_supported