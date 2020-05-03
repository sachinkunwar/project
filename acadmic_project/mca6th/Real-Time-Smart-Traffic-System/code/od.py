#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:26:59 2019

@author: sachin
"""
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image1.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), extract_detected_objects=True)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
