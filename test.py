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

import paddle
from paddle.vision.datasets import MNIST
import attacks
import configs
import helper
import models
import os
import logger
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
for windows add:
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""


def testing(robust_net, blackbox_net, _loss_fn, _testing_dataset, num_restarts=20, net_name=""):
    # 加载robust net
    robust_path = os.path.join(robust_model_folder,
                               "{} with PGD adversarial training.pdparams".format(robust_net.name))
    logger.log_print("load robust network from {}".format(robust_path))
    robust_checkpoint = paddle.load(robust_path)
    robust_net.set_state_dict(robust_checkpoint["trained_net"])

    # 加载blackbox net
    blackbox_path = os.path.join(black_folder,
                                 "{} with PGD adversarial training.pdparams".format(blackbox_net.name))
    logger.log_print("load blackbox network from {}".format(blackbox_path))
    blackbox_checkpoint = paddle.load(blackbox_path)
    blackbox_net.set_state_dict(blackbox_checkpoint["trained_net"])
    blackbox_fgsm_hp = blackbox_checkpoint["fgsm_hp"]
    blackbox_pgd_hp = blackbox_checkpoint["pgd_hp"]

    logger.log_print("FGMS attack selected hyperparams: {}".format(str(blackbox_fgsm_hp)))
    logger.log_print("PGD attack selected hyperparams: {}".format(str(blackbox_pgd_hp)))
    robust_net.eval()
    # measure attacks on test (holdout)
    resistance_results = helper.measure_resistance_on_test(robust_net, blackbox_net, _loss_fn, _testing_dataset,
                                                           to_attacks=[(attacks.FGSM, blackbox_fgsm_hp),
                                                                       (attacks.PGD, blackbox_pgd_hp)],
                                                           num_restarts=num_restarts,
                                                           device=device,
                                                           plot_results=False,
                                                           save_figs=False,
                                                           figs_path=plots_folder,
                                                           plots_title=net_name)
    # unpack resistance_results
    test_acc = resistance_results["test_acc"]  # the accuracy without applying any attack
    fgsm_res = resistance_results["%fgsm"]
    pgd_res = resistance_results["%pgd"]
    # print scores:
    logger.log_print("TEST SCORES of {}:".format(net_name))
    logger.log_print("accuracy on test:                         {}".format(test_acc))
    logger.log_print("accuracy on FGSM constructed examples:    {}".format(fgsm_res))
    logger.log_print("accuracy on PGD constructed examples:     {}".format(pgd_res))


if __name__ == '__main__':
    # initialization
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default="MNIST",
                        help='choose one of: [MNIST, traffic_signs]')
    parser.add_argument('--method', type=str, default='white',
                        help='choose testing mothed: [white, blackA, blackB]')
    parser.add_argument('--num_restarts', type=int, default=20,
                        help='The number of random restart when executing PGD attack.')

    args = parser.parse_args()
    dataset_name = args.dataset_name  # choose from [MNIST, traffic_signs]
    network_architecture = models.CNN_MNIST_OriginNet

    # load configs from configs.py
    experiment_configs = configs.configs_dict[dataset_name]["configs"]
    experiment_hps_sets = configs.configs_dict[dataset_name]["hps_dict"]
    experiment_results_folder = os.path.join(configs.test_result_folder, dataset_name)
    robust_model_folder = os.path.join(configs.checkpoints_folder, "MNIST_Robust_Model")

    if args.method == 'blackA':
        black_folder = os.path.join(configs.checkpoints_folder, "MNIST_BlackboxA")
    elif args.method == 'blackB':
        black_folder = os.path.join(configs.checkpoints_folder, "MNIST_BlackboxB")
    else:
        black_folder = os.path.join(configs.checkpoints_folder, "MNIST_Robust_Model")
        experiment_hps_sets["PGD_attack"]["steps"] = [40]
        experiment_hps_sets["PGD_train"]["steps"] = [40]

    logger_path = os.path.join(experiment_results_folder, "log.txt")
    plots_folder = os.path.join(experiment_results_folder, "plots")

    if not os.path.exists(configs.test_result_folder):
        os.mkdir(configs.test_result_folder)
    if not os.path.exists(experiment_results_folder):
        os.mkdir(experiment_results_folder)

    # set logger
    logger.init_log(logger_path)
    logger.log_print("Dataset name: {}".format(dataset_name))
    logger.log_print("robust model checkpoints folder: {}".format(robust_model_folder))
    logger.log_print("blackbox testing model checkpoints folder: {}".format(black_folder))
    logger.log_print("PGD parameters: ")
    logger.log_print("PGD step: {}    PGD restarts: {}".format(
        experiment_hps_sets["PGD_attack"]["steps"], args.num_restarts))
    logger.log_print("testing results folder: {}".format(experiment_results_folder))

    # 开启0号GPU
    use_gpu = True
    device = paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    logger.log_print("execution device: {}".format(device))

    # get datasets
    _testing_dataset = MNIST(mode='test', download=True,
                             transform=paddle.vision.transforms.Compose([paddle.vision.transforms.ToTensor()]))

    # loss and general training componenets:
    _loss_fn = experiment_configs["loss_function"]

    # testing
    logger.new_section()
    net_name = network_architecture.name
    logger.log_print("Testing on {} box".format(args.method))
    if args.method == 'white':
        robust_net = network_architecture()
        logger.log_print("Network architecture:")
        logger.log_print(str(robust_net))
        testing(robust_net, robust_net, _loss_fn, _testing_dataset,
                num_restarts=args.num_restarts,
                net_name=net_name,
                )
    elif args.method == 'blackA':
        robust_net = network_architecture()
        blackbox_net = network_architecture()
        logger.log_print("robust Network architecture:")
        logger.log_print(str(robust_net))
        logger.log_print("blackbox Network architecture:")
        logger.log_print(str(blackbox_net))
        testing(robust_net, blackbox_net, _loss_fn, _testing_dataset,
                num_restarts=args.num_restarts,
                net_name=net_name,
                )
    else:
        robust_net = network_architecture()
        blackbox_net = models.CNN_MNIST_B()
        logger.log_print("robust Network architecture:")
        logger.log_print(str(robust_net))
        logger.log_print("blackbox Network architecture:")
        logger.log_print(str(blackbox_net))
        testing(robust_net, blackbox_net, _loss_fn, _testing_dataset,
                num_restarts=args.num_restarts,
                net_name=net_name,
                )
