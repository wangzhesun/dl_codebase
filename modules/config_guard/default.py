import os
import sys

from yacs.config import CfgNode as CN

# ----------------------------
# | Start Default Config
# ----------------------------

# Root Config Node
_C = CN()
_C.name = "Experiment Name"
_C.seed = 1221
_C.task = "classification"
_C.num_classes = -1
_C.input_dim = (32, 32)
_C.save_model = False

# DL System Setting
_C.SYSTEM = CN()
_C.SYSTEM.use_cpu = False
_C.SYSTEM.pin_memory = True
_C.SYSTEM.num_gpus = 1		# Number of GPUs to use
_C.SYSTEM.num_workers = 4	# Number of CPU workers for errands

# Backbone
_C.BACKBONE = CN()
_C.BACKBONE.network = "dropout_lenet"
_C.BACKBONE.pretrained_weights = "none"
_C.BACKBONE.forward_need_label = False
_C.BACKBONE.AUTO_ENCODER = CN()
_C.BACKBONE.AUTO_ENCODER.conditional = False

# Classification Layer
_C.CLASSIFIER = CN()
_C.CLASSIFIER.classifier = "none"
_C.CLASSIFIER.factor = 0
_C.CLASSIFIER.bias = False

# Loss
_C.LOSS = CN()
_C.LOSS.loss = "none"
_C.LOSS.loss_factor = 0

# Dataset Settings
_C.DATASET = CN()
_C.DATASET.dataset = 'cifar10'
_C.DATASET.TRANSFORM = CN()
_C.DATASET.TRANSFORM.train_transforms = ('none')
_C.DATASET.TRANSFORM.test_transforms = ('none')
_C.DATASET.TRANSFORM.TRANSFORM_DETAILS = CN()

# Training Settings
_C.TRAIN = CN()
_C.TRAIN.log_interval = 10
_C.TRAIN.batch_size = 64
_C.TRAIN.initial_lr = 0.01
_C.TRAIN.lr_scheduler = 'none'
_C.TRAIN.step_down_gamma = 0.1
_C.TRAIN.step_down_on_epoch = []
_C.TRAIN.max_epochs = 100
_C.TRAIN.optimizer = 'none'

# Validation Settings
_C.VAL = CN()

# Test Settings
_C.TEST = CN()
_C.TEST.batch_size = 256

# ---------------------------
# | End Default Config
# ---------------------------

def update_config_from_yaml(cfg, args):
    '''
    Update yacs config using yaml file
    '''
    cfg.defrost()

    cfg.merge_from_file(args.cfg)

    cfg.freeze()

if __name__ == "__main__":
    # debug print
    print(_C)
