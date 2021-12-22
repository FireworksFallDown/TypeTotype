import cv2
import sys
import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from my_utils import *


def loadYOLO_oneImg(imgPath, imgName: str, labelsTxtPath: str):
    name, extension = os.path.splitext(imgName)

    labelTxt = os.path.join(labelsTxtPath, name + '.txt')

    imgFile = os.path.join(imgPath, imgName)

    assert os.path.exists(labelTxt), "can not find the label file !"
    assert os.path.exists(imgFile), "can not find the image file!"

    imgH, imgW, _ = cv2.imread(imgFile).shape

    labels = []
    with open(labelTxt, "r") as f:
        for line in f.readlines():
            cateId, center_x, center_y, w, h = line.split()
            cateId = int(cateId)
            center_x = float(center_x) * imgW
            center_y = float(center_y) * imgH
            w = float(w) * imgW
            h = float(h) * imgH

            x_min = center_x - w / 2
            y_min = center_y - h / 2
            x_max = x_min + w
            y_max = y_min + h

            labels.append([x_min, y_min, x_max, y_max, cateId])
    return labels


def loadCOCO_oneImg(imgPath, imgName: str, labelJsonPath: str):
    imgFile = os.path.join(imgPath, imgName)

    assert os.path.exists(labelJsonPath), "can not find the label file !"
    assert os.path.exists(imgFile), "can not find the image file!"

    jsonInfo = json.load(open(labelJsonPath))

    imgId2Name = {}
    for imgInfo in jsonInfo['images']:
        imgId2Name[imgInfo['id']] = imgInfo['file_name']

    labels = []

    for ann in jsonInfo['annotations']:
        if imgId2Name[ann['image_id']] == imgName:
            bbox = ann['bbox']
            cocoCateId = ann['category_id']
            xyxy = coco_xywh2xyxy(bbox)

            labels.append(xyxy + [cocoCateId - 1])

    return labels


def loadVOC_oneImg(imgPath, imgName: str, labelTxtPath: str):
    name, extension = os.path.splitext(imgName)
    xmlFile = os.path.join(labelTxtPath, name + '.xml')

    imgFile = os.path.join(imgPath, imgName)

    assert os.path.exists(xmlFile), "can not find the label file : " + xmlFile
    assert os.path.exists(imgFile), "can not find the image file : " + imgFile

    imgH, imgW, _ = cv2.imread(imgFile).shape

    tree = ET.parse(xmlFile)
    root = tree.getroot()

    labels = []

    for child in root:
        if child.tag == 'object':
            # one object
            cateName = child.find('name').text
            xyxy = [float(it.text) for it in child.find('bndbox')]

            labels.append(xyxy + [cateName])
    return labels


def detectionImgShow(imgPath: str, imgName: str, labels: list, CLASSES: list):
    """
    :param img:
    :param labels: list, [x_min,y_min,x_max,y_max,category_id,score(if test mode)]
    :param CLASSES:
    :return:
    """
    img = cv2.imread(os.path.join(imgPath, imgName))

    imgH, imgW, _ = img.shape

    text_scale = imgH / 500 * 0.5
    thinkness = max(int(imgH / 500 * 1), 1)

    cls_num = len(CLASSES)
    clsId2colors = []
    clsName2Colors = {}
    for i in range(cls_num):
        color = ((np.random.random((1, 3)) * 0.6 + 0.4) * 255).tolist()[0]
        clsId2colors.append(color)
        clsName2Colors[CLASSES[i]] = color

    for label in labels:
        if isinstance(label[4], str):
            color = clsName2Colors[label[4]]
            cateName = label[4]
        else:
            color = clsId2colors[int(label[4])]
            cateName = CLASSES[int(label[4])]
        left_top = (int(label[0]), int(label[1]))
        right_bottom = (int(label[2]), int(label[3]))
        text_loc = (int(label[0]), int(label[1]) - 5)
        img = cv2.rectangle(img, left_top, right_bottom, color)
        cv2.putText(img, cateName, text_loc, cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, thinkness)
        if len(label) == 6:
            score = label[-1]
            # draw text of score
    return img
