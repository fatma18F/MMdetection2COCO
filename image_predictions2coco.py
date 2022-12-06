import cv2
import io
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import os, time, random
from string import Template
from shutil import copyfile
import random
from collections import defaultdict
import json
import json
import argparse
import imagesize
from mmdetection.mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv

dir_path='/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images/airplane/day_airplane_0/'

#%cd mmdetection/
def dataset2coco(id, bboxes, scores, bbclasses, confthre,root):
    global annotion_id

    for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(bbclasses[i])+1
            score = scores[i]
            if score < confthre:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            # print(x0,y0)
            w=x1-x0
            h=y1-y0
            image_annotations={ 
                                    "id": int(annotion_id),
                                    "image_id": (id),
                                    "category_id": int(cls_id),
                                    "bbox": [x0, y0, w, h],
                                    "score":float(score)
                                  }
            root["annotations"].append(image_annotations)

            annotion_id+=1
    return root

def convert_category_annotations(orginal_category_info):
    
    categories = []
    num_categories = len(orginal_category_info)
    for i in range(num_categories):
        cat = {}
        cat['id'] = i + 1
        cat['name'] = orginal_category_info[i]
        categories.append(cat)
    
    return categories
#convert_category_annotations(COCO_CLASSES)
def save_annot_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json) 
COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

def infer(path,config_file,checkpoint_file ):
   # Specify the path to model config and checkpoint file
   
   model  = init_detector(config_file, checkpoint_file, device='cuda:0')
   dir_path=path
   annotion_id=0
      
   annotations_json = {
        "categories":convert_category_annotations(COCO_CLASSES),
        "images": [],
        "annotations":[]
   }
   for i , file in enumerate(os.listdir(dir_path)):
      #id=i+1
      filename = os.path.join(dir_path, file)
      if filename[-4:]=='.jpg' or filename[-4:]=='.png':
        w, h = imagesize.get(filename)
        image = {
          "id": int(i),
          "width": float(w),
          "height": float(h),
          "file_name": str(file),
        }
    
        annotations_json["images"].append(image)
        #img=plt.imread(filename)
        pred,seg = inference_detector(model, filename)
        boxes, scores, labels = (list(), list(), list())

        for k, cls_result in enumerate(pred):
        #             print("cls_result", cls_result)
            if cls_result.size != 0:
                if len(labels)==0:
                    boxes = np.array(cls_result[:, :4])
                    scores = np.array(cls_result[:, 4])
                    labels = np.array([k]*len(cls_result[:, 4]))
                else:    
                    boxes = np.concatenate((boxes, np.array(cls_result[:, :4])))
                    scores = np.concatenate((scores, np.array(cls_result[:, 4])))
                    labels = np.concatenate((labels, [k]*len(cls_result[:, 4])))
    
        #img = cv2.imread(filename)
        confthre=0.4
        #out_image = draw_yolox_predictions(img, boxes, scores, labels, confthre, COCO_CLASSES)
        #out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
        #display(Image.fromarray(out_image))
        pred_annotations = dataset2coco(i,boxes,scores,labels,confthre,annotations_json)
        if i ==10:
            break 
   #return annot  
   save_annot_json(pred_annotations, f"OID_predictions.json")

annotion_id=0

config_file = 'mmdetection/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
checkpoint_file = 'mmdetection/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
infer('mmdetection/demo/', config_file, checkpoint_file )
