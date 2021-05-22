import os
import numpy as np
from colorama import Back, Fore
from config import cfg
from dataset import detection_set
from dataset.voc.pascal_voc import PascalVoc
from dataset.coco.coco import COCO


def get_dataset(dataset_sequence, params, mode='train', only_classes=False):
    only_cls_str = 'classes for ' if only_classes else ''
    print(Back.WHITE + Fore.BLACK + 'Loading {}image dataset...'.format(only_cls_str))
    dataset_name = dataset_sequence.split('_')[0]
    if dataset_name == 'detect':
        dataset = detection_set.DetectionSet(params)
        short_name = 'det_set'
        print('Loaded Detection dataset.')
    elif dataset_name == 'voc':
        year = dataset_sequence.split('_')[1]
        image_set = dataset_sequence[(len(dataset_name) + len(year) + 2):]
        if 'devkit_path' in params:
            params['devkit_path'] = os.path.join(cfg.DATA_DIR, params['devkit_path'])
        else:
            print(Back.YELLOW + Fore.BLACK + 'WARNING! '
                  + 'Cannot find "devkit_path" in additional parameters. '
                  + 'Try to use default path (./data/VOCdevkit)...')
            params['devkit_path'] = os.path.join(cfg.DATA_DIR, 'VOCdevkit'+year)
        dataset = PascalVoc(image_set, year, params, only_classes)
        short_name = dataset_name + '_' + year
        print('Loaded {} PascalVoc {} {} dataset.'.format(only_cls_str, year, image_set))
    elif dataset_name == 'coco':
        year = dataset_sequence.split('_')[1]
        image_set = dataset_sequence[(len(dataset_name) + len(year) + 2):]
        if 'data_path' in params:
            params['data_path'] = os.path.join(cfg.DATA_DIR, params['data_path'])
        else:
            print(Back.YELLOW + Fore.BLACK + 'WARNING! '
                  + 'Cannot find "data_path" in additional parameters. '
                  + 'Try to use default path (./data/COCO)...')
            params['data_path'] = os.path.join(cfg.DATA_DIR, 'COCO')
        dataset = COCO(image_set, year, params, only_classes)
        short_name = dataset_name + '_' + year
        print('Loaded {}COCO {} {} dataset.'.format(only_cls_str, year, image_set))
    else:
        raise NotImplementedError(Back.RED + 'Not implement for "{}" dataset!'.format(dataset_name))

    if not only_classes:
        if mode == 'train' and cfg.TRAIN.USE_FLIPPED:
            print(Back.WHITE + Fore.BLACK + 'Appending horizontally-flipped '
                                          + 'training examples...')
            dataset = _append_flipped_images(dataset)
            print('Done.')


        if mode == 'train':
            print(Back.WHITE + Fore.BLACK + 'Filtering image data '
                                          + '(remove images without boxes)...')
            dataset = _filter_data(dataset)
            print('Done.')

    return dataset, short_name


def _append_flipped_images(dataset):
    for i in range(len(dataset)):
        img = dataset.image_data[i].copy()
        img['index'] = len(dataset)
        img['id'] += '_f'
        img['flipped'] = True
        boxes = img['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = img['width'] - oldx2 - 1
        boxes[:, 2] = img['width'] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        img['boxes'] = boxes
        """ do the same for selective search boxes """
        ss_boxes = img['ss_boxes'].copy()
        oldx1 = ss_boxes[:, 0].copy()
        oldx2 = ss_boxes[:, 2].copy()
        ss_boxes[:, 0] = img['width'] - oldx2 - 1
        ss_boxes[:, 2] = img['width'] - oldx1 - 1
        assert (ss_boxes[:, 2] >= ss_boxes[:, 0]).all()
        img['ss_boxes'] = ss_boxes
        dataset.image_data.append(img)
        dataset._image_index.append(img['id'])

    return dataset


def _filter_data(dataset):
    print('Before filtering, there are %d images...' % (len(dataset)))
    i = 0
    while i < len(dataset):
        if len(dataset.image_data[i]['boxes']) == 0:
            del dataset.image_data[i]
            i -= 1
        i += 1

    print('After filtering, there are %d images...' % (len(dataset)))
    return dataset

