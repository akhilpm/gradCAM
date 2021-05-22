import os
import re
import pickle
import uuid
import cv2 as cv
import scipy.io
import json
import numpy as np
import xml.etree.ElementTree as ET
import dataset.utils as utils
from dataset.image_dataset import ImageDataset
from config import cfg

_CLASSES = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

class PascalVoc(ImageDataset):
    def __init__(self, image_set, year, params, only_classes=False):
        ImageDataset.__init__(self, 'voc_' + year + '_' + image_set, params)
        self._image_set = image_set
        self._year = year
        self._devkit_path = params['devkit_path']
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        assert os.path.exists(self._data_path), \
            'Path to data does not exist: {}'.format(self._data_path)
        self._classes = _CLASSES
        if not only_classes:
            self._class_index = dict(zip(self.classes, range(self.num_classes)))
            self._image_index = self._load_image_index()
            #self._image_index = self._image_index[:1002]
            self._image_data = self._load_image_data()
            self._salt = str(uuid.uuid4())
            self._comp_id = 'comp1'

            # PASCAL specific config options
            self.config = {'cleanup': True,
                        'use_salt': False,
                        'use_diff': False,
                        'matlab_eval': False}

    def image_path_at(self, id):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  str(id) + '.jpg')
        assert os.path.exists(image_path), \
            'Image Path does not exist: {}'.format(image_path)
        return image_path


    def _load_image_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = []
            for id in f.readlines():
                _tmp = re.sub(r'\s+', ' ', id).strip().split(' ')
                if len(_tmp) == 1:
                    image_index.append(_tmp[0])
                elif len(_tmp) > 1:
                    if _tmp[1] == '0' or _tmp[1] == '1': image_index.append(_tmp[0])
                else:
                    raise ValueError('Unknown string format: %s' % (id))

        return image_index


    def _load_object_proposals(self, image_data):
        if cfg.TRAIN.PROPOSAL_TYPE == 'SS':
            print("Loading selective search boxes")
            ss_filename = 'voc_'+self._year+'_'+self._image_set+'.mat'
            ss_file_path = os.path.join(self._devkit_path, 'selective_search_data', ss_filename)
            assert os.path.exists(ss_file_path), 'selective search data file does not exist: {}'.format(ss_file_path)
            ss_voc2007_trainval = scipy.io.loadmat(ss_file_path)
            ss_boxes = ss_voc2007_trainval['boxes'][0]
            ss_boxes = [item[:3000] for item in ss_boxes] #select upto 2000 proposals from an image
            for i, data in enumerate(image_data):
                data['ss_boxes'] = ss_boxes[i][:, (1, 0, 3, 2)] - 1
        elif cfg.TRAIN.PROPOSAL_TYPE=='EB':
            print("Loading Edge box proposals")
            eb_filename = 'voc_'+self._year+'_'+self._image_set+'.mat'
            eb_file_path = os.path.join(self._devkit_path, 'edge_box_data', eb_filename)
            assert os.path.exists(eb_file_path), 'Edge box data file does not exist: {}'.format(eb_file_path)
            eb_voc2007_trainval = scipy.io.loadmat(eb_file_path)
            eb_boxes = eb_voc2007_trainval['boxes'][0]
            eb_scores = eb_voc2007_trainval['boxScores'][0]
            eb_boxes = [item[:3000] for item in eb_boxes]
            for i, data in enumerate(image_data):
                data['ss_boxes'] = eb_boxes[i][:, (1, 0, 3, 2)] - 1
                data['box_scores'] = eb_scores[i][:3000]
        else:
            raise ValueError('Proposal type "{}" is not defined!'.format(cfg.TRAIN.PROPOSAL_TYPE))
        return image_data    
        
    def _load_annotation(self, idx, id):
        img_path = self.image_path_at(id)
        img_size = cv.imread(img_path).shape
        file_name = os.path.join(self._data_path, 'Annotations', id + '.xml')
        tree = ET.parse(file_name)
        objects = tree.findall('object')
        objects_count = len(objects)
        
        boxes = np.zeros((objects_count, 4), dtype=np.uint16)
        is_difficult = np.zeros((objects_count), dtype=np.int32)
        is_truncated = np.zeros((objects_count), dtype=np.int32)
        gt_classes = np.zeros((objects_count), dtype=np.int32)
        overlaps = np.zeros((objects_count, self.num_classes), dtype=np.float32)
        areas = np.zeros((objects_count), dtype=np.float32)
        
        for index, obj in enumerate(objects):
            bndbox = obj.find('bndbox')
            # Start coord is 0
            x1 = int(bndbox.find('xmin').text) - 1
            y1 = int(bndbox.find('ymin').text) - 1
            x2 = int(bndbox.find('xmax').text) - 1
            y2 = int(bndbox.find('ymax').text) - 1
            boxes[index, :] = [x1, y1, x2, y2]
            
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            is_difficult[index] = difficult
            
            truncated = obj.find('truncated')
            truncated = 0 if truncated is None else int(truncated.text)
            is_truncated[index] = truncated
            
            cls = self._class_index[obj.find('name').text.lower().strip()]
            gt_classes[index] = cls
            overlaps[index, cls] = 1.0
            areas[index] = (x2 - x1 + 1) * (y2 - y1 + 1)
            
        utils.validate_boxes(boxes, width=img_size[1], height=img_size[0])
        return {'index': idx,
                'id': str(id),
                'path': img_path,
                'width': img_size[1],
                'height': img_size[0],
                'boxes': boxes, 
                'gt_is_difficult': is_difficult, 
                'gt_is_truncated': is_truncated, 
                'gt_classes': gt_classes, 
                'gt_overlaps': overlaps, 
                'gt_areas': areas,
                'flipped': False}


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True