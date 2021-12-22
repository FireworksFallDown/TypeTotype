import numpy
import cv2
import xml.etree.ElementTree as ET
import os
import json
from my_utils import yolo_xywh2coco_xywh, xyxy2coco_xywh


# load VOC to coco type
def loadVOC(classNames: list, imagesPath, xmlPath):
    Annotations = {"images": [], "annotations": [], "categories": []}

    imgNameList = []
    for root, _, files in os.walk(imagesPath):
        imgNameList = files

    imgIdIdx = 0
    imgId2Name = {}
    imgName2Id = {}
    name2ImgName = {}
    for imgName in imgNameList:
        imgId2Name[imgIdIdx] = imgName
        imgName2Id[imgName] = imgIdIdx
        name, extension = os.path.splitext(imgName)
        name2ImgName[name] = imgName
        imgIdIdx += 1

    cateId2Name = {}
    cateName2Id = {}
    for i, cateName in enumerate(classNames):
        cateId2Name[i + 1] = cateName  # coco class start from 1
        cateName2Id[cateName] = i + 1

    xmlList = []
    for root, _, files in os.walk(xmlPath):
        xmlList = files

    annId = 0
    for xmlFile in xmlList:
        name = xmlFile[:-4]
        imgName = name2ImgName[name]
        imgH, imgW, _ = cv2.imread(imagesPath + imgName).shape

        # write image info
        Annotations["images"].append(
            {
                "file_name": imgName,
                "height": imgH,
                "width": imgW,
                "id": imgName2Id[imgName]
            })

        # analyze xml
        tree = ET.parse(os.path.join(xmlPath,xmlFile))
        root = tree.getroot()

        for child in root:
            if child.tag == 'object':
                # one object
                cateName = child.find('name').text
                classId = cateName2Id[cateName]
                # xyxy = [float(it.text) for it in child.find('bndbox')]
                xmin = float(child.findall('xmin').text)
                ymin = float(child.findall('ymin').text)
                xmax = float(child.findall('xmax').text)
                ymax = float(child.findall('ymax').text)
                # for loc in child.find('bndbox'):
                #     if loc.tag == 'xmin':
                #         xmin = float(loc.text)
                #     if loc.tag == 'ymin':
                #         ymin = float(loc.text)
                #     if loc.tag == 'xmax':
                #         xmax = float(loc.text)
                #     if loc.tag == 'ymax':
                #         ymax = float(loc.text)
                x, y, w, h = coco_xywh = xyxy2coco_xywh([xmin,ymin,xmax,ymax])
                # write ann info
                Annotations["annotations"].append(
                    {
                        "segmentation": [],
                        "iscrowd": 0,  # 0 or 1
                        "area": w * h,  # float or double
                        "image_id": imgName2Id[imgName],  # int
                        "bbox": coco_xywh,  # list[float], [x,y,w,h]
                        "category_id": classId,
                        "id": annId  # int
                    }
                )
                annId += 1

    # write categories
    for cateId in cateId2Name:
        cateName = cateId2Name[cateId]
        Annotations["categories"].append({"id": cateId, "name": cateName})

    return Annotations


# load YOLOv5 to coco type
def loadYOLO(classNames: list, imagesPath, labelsPath):
    Annotations = {"images": [], "annotations": [], "categories": []}

    imgNameList = []
    for root, _, files in os.walk(imagesPath):
        imgNameList = files
    labelNameList = []
    for root, _, files in os.walk(labelsPath):
        labelNameList = files

    imgIdIdx = 0
    imgId2Name = {}
    imgName2Id = {}
    name2ImgName = {}
    for imgName in imgNameList:
        imgId2Name[imgIdIdx] = imgName
        imgName2Id[imgName] = imgIdIdx
        name, extension = os.path.splitext(imgName)
        name2ImgName[name] = imgName
        imgIdIdx += 1

    cateId2Name = {}
    for i, cateName in enumerate(classNames):
        cateId2Name[i + 1] = cateName

    annId = 0
    for txtName in labelNameList:
        imgName = name2ImgName[txtName[:-4]]
        imgH, imgW, _ = cv2.imread(imagesPath + imgName).shape

        # write image info
        Annotations["images"].append(
            {
                "file_name": imgName,
                "height": imgH,
                "width": imgW,
                "id": imgName2Id[imgName]
            })

        with open(os.path.join(labelsPath,txtName), "r") as f:
            for line in f.readlines():
                cateId, center_x, center_y, w, h = line.split()
                cateId = int(cateId)
                center_x = float(center_x)
                center_y = float(center_y)
                w = float(w)
                h = float(h)

                coco_xywh = yolo_xywh2coco_xywh([center_x, center_y, w, h], imgW, imgH)

                # write ann info
                Annotations["annotations"].append(
                    {
                        "segmentation": [],
                        "iscrowd": 0,  # 0 or 1
                        "area": coco_xywh[2] * coco_xywh[3],  # float or double
                        "image_id": imgName2Id[imgName],  # int
                        "bbox": coco_xywh,  # list[float], [x,y,w,h]
                        "category_id": cateId + 1,  # yolo class start from 0, coco start from 1
                        "id": annId  # int
                    }
                )
                annId += 1

    # write categories
    for cateId in cateId2Name:
        cateName = cateId2Name[cateId]
        Annotations["categories"].append({"id": cateId, "name": cateName})

    return Annotations


# load COCO
def loadCOCO(cocoJsonFile):
    jsonInfo = json.load(open(cocoJsonFile))
    return jsonInfo
