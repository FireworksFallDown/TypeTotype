import cv2
import os
import argparse
from my_config import CLASSES
import type2type
import my_visualize
# import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="process some args")
    parser.add_argument('--img_path', type=str, help="the images Path of your dataset, eg: /home/yourPath/Images")
    parser.add_argument('--mode', type=str,
                        choices=['VOC2COCO', 'VOC2YOLO', 'YOLO2COCO', 'YOLO2VOC', 'COCO2YOLO', 'COCO2VOC',
                                 'LabelView_COCO', 'LabelView_VOC', 'LabelView_YOLO'],
                        help="the mode of changing label type to other type")
    parser.add_argument('--label_path', type=str, help="the labels Path of your dataset, eg: /home/yourPath/Labels , \
                                                    if your original data is COCO type, this argument should be the \
                                                    json file name, eg: /home/yourPath/label.json ")
    parser.add_argument('--save_path', type=str, default="./changeResult",
                        help="the path to save the final labels, eg: /home/savePath , \
                                                    if your target type is COCO, this argument should be the \
                                                    json file name, eg: /home/yourPath/label_save.json ")

    args = parser.parse_args()
    return args


def label_view(img_path, label_path, mode, classes):
    assert os.path.exists(img_path), "the img_path do not exist!"
    assert os.path.exists(label_path), "the label_path do not exist!"

    mode2ShowMode = {'LabelView_COCO': "COCO", 'LabelView_VOC': "VOC", 'LabelView_YOLO': "YOLO"}

    imgsList = []
    for root, _, files in os.walk(img_path):
        imgsList = files

    for imgName in imgsList:
        show_mode = mode2ShowMode[mode]
        loadFunc = getattr(my_visualize, 'load' + show_mode + '_oneImg')
        labels = loadFunc(img_path, imgName, label_path)
        imgView = my_visualize.detectionImgShow(img_path, imgName, labels, classes)

        cv2.imshow('label_view', imgView)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return



def main():
    arg = get_args()

    classes = CLASSES

    mode = arg.mode
    img_path = arg.img_path
    label_path = arg.label_path
    save_path = arg.save_path

    if mode in ['LabelView_COCO', 'LabelView_VOC', 'LabelView_YOLO']:
        label_view(img_path, label_path, mode, classes)
        return

    changeFunc = getattr(type2type, mode)
    changeFunc(classes, img_path, label_path, save_path)


def startFun(img_path, label_path, mode, save_path='results/'):
    """
    this function is for Qt
    :param img_path:
    :param label_path:
    :param save_path:
    :param mode:
    :return:
    """
    classes = CLASSES

    img_path = os.path.join(img_path, "")
    if mode not in ['COCO2YOLO', 'COCO2VOC','LabelView_COCO']:
        label_path = os.path.join(label_path, "")
    if mode not in ['VOC2COCO', 'YOLO2COCO']:
        save_path = os.path.join(save_path, "")

    if mode in ['LabelView_COCO', 'LabelView_VOC', 'LabelView_YOLO']:
        label_view(img_path, label_path, mode, classes)
        return

    changeFunc = getattr(type2type, mode)
    changeFunc(classes, img_path, label_path, save_path)


if __name__ == '__main__':
    main()
