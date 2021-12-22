"""
COCO type:

Annotations.json
--------------------------------------------------------------------------------------------------
{
    "images": [
        {
            "file_name": "***.jpg", # string
            "height": ***,          # int
            "width": ***,           # int
            "id": *                 # int
        },...]
    "annotations": [
        {
            "iscrowd": 0,           # 0 or 1
            "area": *,              # float or double
            "image_id": 0,          # int
            "bbox": [*, *, *, *],   # list[float], [x,y,w,h]
            "category_id": *,       # int
            "id": *                 # int
        },...]
    "categories": [
        {"id": *, "name": "***"},   # id start from 1
        ...
        ]
}
--------------------------------------------------------------------------------------------------
"""

"""
VOC type:
Annotations
    |—— ImgName_1.xml
    |—— ImgName_2.xml
    ...

--------------------------------------------------------------------------------------------------
ImgName_1.xml :
--------------------------------------------------------------------------------------------------
<?xml version="1.0" encoding="utf-8"?>
<annotation>
    <folder>VOC2007</folder>
    <filename>***.jpeg</filename>
    <size>
        <width>***</width>
        <height>***</height>
        <depth>3</depth>
    </size>
    <object>
        <name>***</name>        # category name
        <bndbox>
            <xmin>*</xmin>
            <xmax>*</xmax>
            <ymin>*</ymin>
            <ymax>*</ymax>
        </bndbox>
        <truncated>0</truncated>
        <difficult>0</difficult>
    </object>
    ...
    <segmented>0</segmented>
</annotation>
--------------------------------------------------------------------------------------------------

"""


"""
YOLO V5 type:
--------------------------------------------------------------------------------------------------
imgName_1.txt:
--------------------------------------------------------------------------------------------------
cateId  center_x  center_y  w  h  # normalization type, eg: 0 0.473667 0.397000 0.116000 0.337000
...
--------------------------------------------------------------------------------------------------
    
"""