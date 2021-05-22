import os
from easydict import EasyDict as edict

cfg = edict()

cfg.GENERAL = edict()
cfg.GENERAL.MIN_IMG_RATIO = 0.5
cfg.GENERAL.MAX_IMG_RATIO = 2.0
cfg.GENERAL.MIN_IMG_SIZE = 600
cfg.GENERAL.MAX_IMG_SIZE = 1000
cfg.GENERAL.POOLING_MODE = 'pool'
cfg.GENERAL.POOLING_SIZE = 7
cfg.GENERAL.SCALES = [480,576,688,864,1200]

cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.LR_DECAY_STEP = 5
cfg.TRAIN.LR_DECAY_GAMMA = 0.1
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.DOUBLE_BIAS = True
cfg.TRAIN.BIAS_DECAY = False
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.USE_FLIPPED = True
#++++++ options added by me ++++++++++++
cfg.TRAIN.PROPOSAL_TYPE = 'SS'
cfg.TRAIN.USE_SS_GTBOXES = True
cfg.TRAIN.PROPAGATE_OVERLAP = 0.3
cfg.TRAIN.MILESTONES = [5, 10]
cfg.TRAIN.NUM_BOXES_PERCLASS = 20
cfg.TRAIN.NUM_PROPOSALS = 2500
cfg.TRAIN.C = 0.0
cfg.TRAIN.ALPHA = 0.9
cfg.TRAIN.SOFTMAX_TEMP = 1.0
cfg.TRAIN.CLUSTER_THRESHOLD = 0.7
cfg.NUM_CLASSES = 21
cfg.VIS_OFFSET = 50
#++++++++++++++++++++++++++++++++++++++
cfg.RPN = edict()
cfg.RPN.ANCHOR_SCALES = [8, 16, 32]
cfg.RPN.ANCHOR_RATIOS = [0.5, 1, 2]
cfg.RPN.FEATURE_STRIDE = 16

cfg.TEST = edict()
cfg.TEST.RPN_PRE_NMS_TOP = 7000
cfg.TEST.RPN_POST_NMS_TOP = 500
cfg.TEST.RPN_NMS_THRESHOLD = 0.7
cfg.TEST.NMS = 0.3
cfg.TEST.MULTI_SCALE_TESTING = False

cfg.RESNET = edict()
cfg.RESNET.NUM_FREEZE_BLOCKS = 1


def update_config_from_file(config_file_path):
    config_path = os.path.join(cfg.ROOT_DIR, config_file_path)
    assert os.path.exists(config_path), 'Config file does not exist'

    import yaml
    with open(config_path, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_configs(yaml_cfg, cfg)


def _merge_configs(a, b):
    assert type(a) is edict, 'Config file must be edict'

    for k, v in a.items():
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        if type(b[k]) is not type(a[k]):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              + 'for config key: {}').format(type(b[k]),
                                                             type(v), k))

        if type(v) is edict:
            try:
                _merge_configs(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v
