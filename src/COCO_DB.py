#!/usr/bin/env python
"""
COCO_DB.py is a class that provides an access to MS COCO dataset
Created 6/16/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


from dataset.coco.PythonAPI.pycocotools.coco import COCO as C
from .util import check_path, assert_list
from os.path import join
from cv2 import imread, cvtColor, COLOR_BGR2RGB

class COCO(object):


    def __init__(self, ANNOTATIONS_FILE_NAME, PATH2IMAGES, CLASSES):


        self.annotations_file = check_path(ANNOTATIONS_FILE_NAME)
        self.IMAGES_PATH = check_path(PATH2IMAGES)
        self.coco = C(self.annotations_file)

        self._imgIds = [] # Set by CLASSES_NEEDED setter
        self._catIds = [] # Set by CLASSES_NEEDED setter

        categories = self.coco.loadCats(self.coco.getCatIds())
        self.CLASSES_AVAILABLE = [category['name'] for category in categories]
        self.CATEGORIES = set([category['supercategory'] for category in categories])
        self.CLASSES_NEEDED = assert_list(CLASSES)

        self.samples_read_counter = 0

    @property
    def CLASSES_NEEDED(self):
        return self._CLASSES_NEEDED

    @CLASSES_NEEDED.setter
    def CLASSES_NEEDED(self, values):
        for value in values:
            assert value in self.CLASSES_AVAILABLE, \
                "CLASSES_AVAILABLE array is incorrect. The following class (from the array) " + \
                "is not present in self.CLASSES_AVAILABLE array: {}.". \
                    format(value)
        self._CLASSES_NEEDED = values

        # build an array of file indicies
        img_ids = []
        cat_ids = []
        for class_name in self._CLASSES_NEEDED:
            cat_id = self.coco.getCatIds(catNms=[class_name])
            cat_ids.extend(cat_id)
            img_ids.extend(self.coco.getImgIds(catIds=cat_id))

        # update image ids
        self._imgIds = img_ids
        self._catIds = cat_ids

    @property
    def samples_read_counter(self):
        return self._samples_read_counter

    @samples_read_counter.setter
    def samples_read_counter(self, value):
        self._samples_read_counter = value if value < self.epoch_size() else 0


    def _provide_img_tags(self, id, coco_labels=True, iscrowd=None):
        """
        Protocol describing the implementation of a method that provides tags for the image file based on
        an image id.

        [{
        "image_id" : int,
        "category_id" : int,
        "bbox" : [x,y,width,height], top left x,y (y=0 is the top) and width, height delta to get to bottom right
        "score" : float,
        }]

        :param id: dataset specific image id
        :param coco_labels(optional): indicates wherher coco label ids should be returned
        default value is False - BATCH_CLASS ids should be returned
        :return: an array containing the list of tags
        """


        # Extract annotation ids
        ann_ids = self.coco.getAnnIds(imgIds=[id], catIds=self._catIds, iscrowd=iscrowd)
        # get all annotations available

        anns = self.coco.loadAnns(ids=ann_ids)

        # parse annotations into a list
        cat_ids = []
        bbox_values = []
        segmentation = []
        for ann in anns:
            if ann['iscrowd']!=1:
                cat_ids.append(ann['category_id'])
                bbox_values.append(ann['bbox'])
                segmentation.append(ann['segmentation'])

        if not(coco_labels):
            cats = self.coco.loadCats(ids=cat_ids)
            cat_ids = [self.CLASSES_NEEDED.index(cat['name']) for cat in cats]

        return cat_ids, bbox_values, segmentation

    def _provide_img_file_name(self, id):
        """
        Protocol describing the implementation of a method that provides the name of the image file based on
        an image id.
        :param id: dataset specific image id
        :return: string containing file name
        """

        descriptions = self.coco.loadImgs(id)[0]
        return descriptions['file_name']


    def epoch_size(self):
        return len(self._imgIds)


    def get_sample(self):

        NotFound = True
        while NotFound:
            # 1. Get img_id
            img_id = self._imgIds[self.samples_read_counter]

            # 2. Get annotatinos
            labels, gtbboxes, segmentation = self._provide_img_tags(img_id)

            self.samples_read_counter+=1

            # 2. Read the file name
            file_name = self._provide_img_file_name(img_id)
            try:
                file_path = check_path(join(self.IMAGES_PATH, file_name))
                im = cvtColor(imread(file_path), COLOR_BGR2RGB)
                NotFound=False
            except:
                pass

        return im, labels, gtbboxes