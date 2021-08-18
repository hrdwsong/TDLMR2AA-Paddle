# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle import nn
import trainer
import os


# paths
data_root_dir = os.path.join(".", "data")
checkpoints_folder = os.path.join(".", "checkpoints")
results_folder = os.path.join(".", "results_folder")
test_result_folder = os.path.join(".", "test_results_folder")

# general configurations:
save_checkpoints = True  # 改
load_checkpoints = False  # To use a saved checkpoint instead re-training. 改
show_attacks_plots = False  # plots cannot be displayed in NOVA
save_attacks_plots = False
show_validation_accuracy_each_epoch = True  # becomes True if using early stopping
imgs_to_show = 4  # maximal number images to show in a grid of images
val_ratio = 0.7


MNIST_experiments_configs = {
    "adversarial_training_stopping_criteria": trainer.ConstantStopping(50),
    "training_stopping_criteria": trainer.ConstantStopping(5),
    "loss_function": nn.CrossEntropyLoss(),  # the nets architectures are built based on CE loss
    "add_natural_examples": False
}

MNIST_experiments_hps = {
    "FGSM_attack": {
        "epsilon": [0.3],
    },

    "PGD_attack": {
        "alpha": [0.01],
        "steps": [100],  # 改100
        "epsilon": [0.3]
    },

    "FGSM_train": {
        "epsilon": [0.3],
    },

    "PGD_train": {
        "alpha": [0.01],
        "steps": [100],  # 改100
        "epsilon": [0.3]
    },

    "nets_training": {
        "lr": [0.0003],
        "batch_size": [128],
        # "lr_scheduler_gamma": [0.95]
    },
}


configs_dict = {
    "MNIST": {
        "configs": MNIST_experiments_configs,
        "hps_dict": MNIST_experiments_hps
    },
}
