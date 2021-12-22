import json
import numpy
import cv2
import xml.etree.ElementTree as ET
import os


# add blank to xml root for pleasing to the eye
def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# yolo bbox to coco bbox
def yolo_xywh2coco_xywh(yolo_xywh: list, imgW: int, imgH: int):
    center_x = yolo_xywh[0] * imgW
    center_y = yolo_xywh[1] * imgH
    w = yolo_xywh[2] * imgW
    h = yolo_xywh[3] * imgH

    x_min = center_x - w / 2
    y_min = center_y - h / 2

    coco_xywh = [x_min, y_min, w, h]

    return coco_xywh


# coco bbox to yolo bbox
def coco_xywh2yolo_xywh(coco_xywh: list, imgW: int, imgH: int):
    center_x = coco_xywh[0] + coco_xywh[2] / 2
    center_y = coco_xywh[1] + coco_xywh[3] / 2

    yolo_xywh = str(center_x / imgW) + ' ' + str(center_y / imgH) + ' ' + \
                str(coco_xywh[2] / imgW) + ' ' + str(coco_xywh[3] / imgH)

    return yolo_xywh


def xyxy2coco_xywh(xyxy: list):
    w = xyxy[2]-xyxy[0]
    h = xyxy[3]-xyxy[1]

    x_min = xyxy[0]
    y_min = xyxy[1]

    coco_xywh = [x_min, y_min, w, h]

    return coco_xywh


def coco_xywh2xyxy(coco_xywh: list):
    w = coco_xywh[2]
    h = coco_xywh[3]

    x_min = coco_xywh[0]
    y_min = coco_xywh[1]
    x_max = x_min + w
    y_max = y_min + h

    xyxy = [x_min, y_min, x_max, y_max]

    return xyxy


def saveCOCOJson(Annotations,saveJson):
    json_str = json.dumps(Annotations, indent=4)
    with open(saveJson, 'w') as json_file:
        json_file.write(json_str)


# xml write img info
def generateImgInfo(imagesPath, imgName, imgH, imgW, imgDepth):
    # generate the xml content
    root = ET.Element('annotation')
    root.text = '\n\t'
    # folder
    folder = ET.SubElement(root, 'folder')
    folder.text = imagesPath
    folder.tail = '\n\t'
    # filename
    filename = ET.SubElement(root, 'filename')
    filename.text = imgName
    filename.tail = '\n\t'
    # source
    source = ET.SubElement(root, 'source')
    source.text = '\n\t'
    database = ET.SubElement(source, 'database')
    database.text = 'not important'
    database.tail = '\n\t'
    annotation = ET.SubElement(source, 'annotation')
    annotation.text = 'not important'
    annotation.tail = '\n\t'
    image = ET.SubElement(source, 'image')
    image.text = 'not important'
    image.tail = '\n\t'
    flickrid = ET.SubElement(source, 'flickrid')
    flickrid.text = 'not important'
    flickrid.tail = '\n\t'

    # owner
    owner = ET.SubElement(root, 'owner')
    owner_flickrid = ET.SubElement(owner, 'flickrid')
    owner_flickrid.text = 'not important'
    owner_name = ET.SubElement(owner, 'name')
    owner_name.text = 'WHO'

    # img size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(imgW)
    height = ET.SubElement(size, 'height')
    height.text = str(imgH)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(imgDepth)

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'  # only for object detection

    indent(root)

    tree = ET.ElementTree(root)
    return tree


# xml insert object subElement
def insertObject(tree, cateName, bbox):
    """
    <object>
        <name>className</name>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>0</xmin>
            <ymin>0</ymin>
            <xmax>0</xmax>
            <ymax>0</ymax>
        </bndbox>
    </object>
    """

    # assert os.path.exists(xmlPath)
    #
    # tree = ET.parse(xmlPath)
    root = tree.getroot()

    object = ET.SubElement(root, 'object')

    name = ET.SubElement(object, 'name')
    name.text = cateName
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = '0'

    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(bbox[0])
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(bbox[1])
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(bbox[0] + bbox[2])
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(bbox[1] + bbox[3])

    indent(root)

    tree = ET.ElementTree(root)

    return tree

