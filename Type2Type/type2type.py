import numpy
import cv2
import xml.etree.ElementTree as ET
import os
import json
from my_utils import *
from my_loadData import *


# <LBR>: turn the COCO type to YOLOv5 and save to YOLOv5 type
def turnCOCO2YOLO(imagesPath,jsonInfo,savePath):

    imgId2Name = {}
    for imgInfo in jsonInfo['images']:
        imgId2Name[imgInfo['id']] = imgInfo['file_name']

    # the category id start from 0 in YOLOv5, but in COCO is 1
    cateCocoId2yoloId = {}
    for cate in jsonInfo['categories']:
        cateCocoId2yoloId[cate['id']] = cate['id'] - 1

    # <LBR>: should ensure the labels.txt not exist
    for imgInfo in jsonInfo['images']:
        name, extension = os.path.splitext(imgInfo['file_name'])
        txtName = os.path.join(savePath, name + '.txt')
        assert not os.path.exists(txtName), "you must ensure there are no any txt files in your savePath: " + savePath

    for ann in jsonInfo['annotations']:
        imgName = imgId2Name[ann['image_id']]
        name, extension = os.path.splitext(imgName)
        txtName = os.path.join(savePath, name + '.txt')

        bbox = ann['bbox']
        cocoCateId = ann['category_id']

        img = cv2.imread(os.path.join(imagesPath, imgName))
        imgH = img.shape[0]
        imgW = img.shape[1]

        yoloCateId = cateCocoId2yoloId[cocoCateId]
        yolo_xywh = coco_xywh2yolo_xywh(bbox, imgW, imgH)

        label = str(yoloCateId) + ' ' + yolo_xywh + '\n'

        file = open(txtName, 'a')
        file.write(label)
        file.close()


# <LBR>: turn the COCO type to YOLOv5 and save to YOLOv5 type
def turnCOCO2VOC(imagesPath, jsonInfo, savePath):

    imgId2Name = {}
    for imgInfo in jsonInfo['images']:
        imgId2Name[imgInfo['id']] = imgInfo['file_name']

    cateId2Name = {}
    for cate in jsonInfo['categories']:
        cateId2Name[cate['id']] = cate['name']

    for imgInfo in jsonInfo['images']:
        imgName = imgInfo['file_name']
        name, extension = os.path.splitext(imgName)

        img = cv2.imread(os.path.join(imagesPath, imgName))
        imgH, imgW, imgDepth = img.shape

        xmlSavePath = os.path.join(savePath, name + '.xml')

        tree = generateImgInfo(imagesPath, imgName, imgH, imgW, imgDepth)

        tree.write(xmlSavePath, encoding='utf-8', xml_declaration=True, method='xml')

    # write object info
    for ann in jsonInfo['annotations']:
        imgName = imgId2Name[ann['image_id']]
        name, extension = os.path.splitext(imgName)
        xmlName = os.path.join(savePath, name + '.xml')

        bbox = ann['bbox']
        cateName = cateId2Name[ann['category_id']]

        assert os.path.exists(xmlName)

        tree = ET.parse(xmlName)
        tree = insertObject(tree, cateName, bbox)

        tree.write(xmlName, encoding='utf-8', xml_declaration=True, method='xml')


#   ------------------------------------------------------------------------------------    #


# <LBR>: COCO2YOLOv5
def COCO2YOLO(classNames,imagesPath, cocoJsonFile, savePath):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    jsonInfo = loadCOCO(cocoJsonFile)
    turnCOCO2YOLO(imagesPath,jsonInfo,savePath)


# <LBR>: COCO2VOC
def COCO2VOC(classNames,imagesPath, cocoJsonFile, savePath):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    jsonInfo = loadCOCO(cocoJsonFile)
    turnCOCO2VOC(imagesPath,jsonInfo,savePath)


# <LBR>: VOC to COCO
def VOC2COCO(classNames,imagesPath,xmlsPath,savePath):
    jsonInfo = loadVOC(classNames,imagesPath,xmlsPath)
    saveCOCOJson(jsonInfo,savePath)


# <LBR>: VOC to YOLOv5
def VOC2YOLO(classNames,imagesPath,xmlsPath,savePath):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    jsonInfo = loadVOC(classNames, imagesPath, xmlsPath)
    turnCOCO2YOLO(imagesPath,jsonInfo,savePath)



# <LBR>: YOLO To COCO
def YOLO2COCO(classNames,imagesPath,labelsPath,savePath):
    jsonInfo = loadYOLO(classNames,imagesPath,labelsPath)
    saveCOCOJson(jsonInfo,savePath)


# <LBR>: YOLO to VOC
def YOLO2VOC(classNames,imagesPath,labelsPath,savePath):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    jsonInfo = loadYOLO(classNames, imagesPath, labelsPath)
    turnCOCO2VOC(imagesPath,jsonInfo,savePath)

