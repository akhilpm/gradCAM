import numpy as np
import os
import torch
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from model.clf_net import Cls_Net
from model.gap_cls_net import GAP_Net
from model.gradCAM import gradCAM
from model.CAM import CAM
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import cv2 as cv


watch_list = ['000012', '000017', '000019', '000021', '000026', '000036', '000089', '000102', '000121', '000130', '000198']

def cam_test(dataset, net, load_dir, session, epoch, log, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    #fetch dataset
    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_test)
    model_path = os.path.join(cfg.DATA_DIR, load_dir, net, ds_name, 'cam_{}_{}.pth'.format(session, epoch))
    log.info(Back.WHITE + Fore.BLACK + 'Loading model from %s' % (model_path))
    checkpoint = torch.load(model_path, map_location=device)
    cam_model = Cls_Net(dataset.num_classes-1)
    #cam_model = GAP_Net(dataset.num_classes-1)
    cam_model.to(device)
    cam_model.load_state_dict(checkpoint['model'])
    cam = gradCAM(cam_model)
    #cam = CAM(cam_model)

    save_root_dir = os.path.join(cfg.DATA_DIR, 'debug', 'session_' + str(session))
    save_dir = os.path.join(save_root_dir, 'gt_heatmap')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_size = len(dataset)
    cam_model.eval()
    for step, data in enumerate(loader):
        image_data = data[0].to(device)
        image_info = data[1].to(device)
        image_labels = data[3]
        image_ids = data[4]
        real_gt_boxes = data[5].to(device)
        if step%200==0:
            watch_list.append(image_ids[0])
            print("Step: {}".format(step))
        if image_ids[0] in watch_list:
            height, width = image_data.size(2),  image_data.size(3)
            image = cv.imread(dataset.image_path_at(image_ids[0]))
            image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
            for i, label in enumerate(image_labels[0]):
                saliency, logits = cam(image_data, label)
                max_act = np.max(saliency)
                binary_mask = (saliency>=0.2*max_act)
                binary_mask = np.uint8(255 * binary_mask)
                contours, hierarchy = cv.findContours(cv.UMat(binary_mask), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
                if width > height:
                    fig, ax = plt.subplots(2, 1)
                else:
                    fig, ax = plt.subplots(1, 2)
                ax[0].axis('off')
                ax[1].axis('off')
                image_rgb = cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB)
                cv.drawContours(image_rgb, contours, -1, (0, 255, 0), 3)
                cntsSorted = sorted(contours, key=lambda x: cv.contourArea(x))
                for cntr in cntsSorted:
                    x, y, w, h = cv.boundingRect(cntr)
                    cv.rectangle(image_rgb, (x, y), (x + w, y + h), (0,255,0), 3)
                ax[0].imshow(image_rgb)
                ax[1].imshow(binary_mask, cmap='jet', interpolation='nearest')
                class_name = dataset.classes[label]
                save_path = os.path.join(save_dir, image_ids[0] + '_' + class_name + '_epoch_' + str(epoch) + '_map.jpg')
                fig.savefig(save_path)
                fig.clf()
                plt.close('all')