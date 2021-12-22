# TypeTotype
Change the target detection dataset type to another type

This program can help you to change the type of your target detection Dataset to another type, and also can help you view visualization of labels, for example, 
change COCO dataset to VOC type, change VOC to YOLO type...

### support mode:
1. VOC to COCO
2. VOC to YOLO
3. COCO to VOC
4. COCO to YOLO
5. YOLO to COCO
6. YOLO to VOC
7. COCO label view
8. VOC label view
9. YOLO label view


### How to use:

**request**
ensure those package: opencv-python, json, xml, numpy installed in your environment

① **you should modify the CLASSES in the `my_config.py` to the categories of your dataset**

② then you can run this command line under the Type2Type directory

`python main.py --img_path [your path] --mode [your mode] --label_path [your label path] --save_path [your save path]`


### NOTES:
① you can run `python main.py --help` for help

② the [mode] must be one of [VOC2COCO, VOC2YOLO, YOLO2COCO, YOLO2VOC, COCO2YOLO, COCO2VOC,LabelView_COCO, LabelView_VOC, LabelView_YOLO]

③ you can ignore the save_path if you only want to view label

④ For Windows, I also offer an application which named Type2Type.exe, you can run the Type2Type.exe
like this:
<br>
![image](https://user-images.githubusercontent.com/96516755/147052687-13270e97-9391-4de3-a521-4e1817e969bd.png)
<br>
you can select your path and mode, then press the START.

⑤ all of your path and file name should be English, otherwise the program can't work
