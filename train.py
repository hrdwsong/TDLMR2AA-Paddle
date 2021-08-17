# -*- coding: utf-8 -*-
# coding: utf-8
import time
import paddle
from paddle.vision.datasets import MNIST
import attacks
import configs
import datasets
import helper
import models
import trainer
import os
import shutil
import logger
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
for windows add:
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""


def start_training(net, _loss_fn, _training_dataset, _testing_dataset, epochs, net_name="", train_attack=None,
                   attack_training_hps_gen=None, load_checkpoint=False, save_checkpoint=False, show_plots=False,
                   save_plots=False, show_validation_accuracy_each_epoch=False):
    hps_gen = net_training_hps_gen
    if train_attack is not None:
        hps_gen = helper.concat_hps_gens(net_training_hps_gen, attack_training_hps_gen)

    # apply hyperparameters-search to get a trained network
    net_state_dict, net_hp = helper.full_train_of_nn_with_hps(net, _loss_fn, _training_dataset,
                                                              hps_gen, epochs, device=device,
                                                              train_attack=train_attack,
                                                              show_validation=show_validation_accuracy_each_epoch,
                                                              add_natural_examples=experiment_configs[
                                                                  "add_natural_examples"])
    net.set_state_dict(net_state_dict)
    # net.eval()  # from now on we only evaluate net.

    logger.log_print("training selected hyperparams: {}".format(str(net_hp)))

    # attack selected net using FGSM:
    fgsm_hp, fgsm_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _testing_dataset,
                                                                    fgsm_attack_hps_gen, net_hp, attacks.FGSM,
                                                                    device=device, plot_results=False,
                                                                    save_figs=False, figs_path=plots_folder)
    # logger.log_print("FGSM attack selected hyperparams: {}".format(str(fgsm_hp)))

    # attack selected net using PGD:
    pgd_hp, pgd_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _testing_dataset,
                                                                  pgd_attack_hps_gen, net_hp, attacks.PGD,
                                                                  device=device, plot_results=False,
                                                                  save_figs=False,
                                                                  figs_path=plots_folder)
    # logger.log_print("PGD attack selected hyperparams: {}".format(str(pgd_hp)))

    # measure attacks on test
    resistance_results = helper.measure_resistance_on_test(net, net, _loss_fn, _testing_dataset,
                                                           to_attacks=[(attacks.FGSM, fgsm_hp),
                                                                       (attacks.PGD, pgd_hp)],
                                                           device=device,
                                                           plot_results=show_plots,
                                                           save_figs=save_plots,
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

    # save checkpoint
    res_dict = {
        "trained_net": net,
        "net_hp": net_hp,
        "fgsm_hp": fgsm_hp,
        "pgd_hp": pgd_hp,
        "resistance_results": resistance_results
    }

    if save_checkpoint and not load_checkpoint:
        to_save_res_dict = res_dict
        to_save_res_dict["trained_net"] = net.state_dict()
        checkpoint_path = os.path.join(experiment_checkpoints_folder, "{}.pdparams".format(net_name))
        logger.log_print("save network to {}".format(checkpoint_path))
        paddle.save(to_save_res_dict, checkpoint_path)

    return res_dict


# Build robust network with PGD and FGSM and compare them
def build_robust_network(net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                         net_name="", load_checkpoint=False, save_checkpoint=False, show_plots=False, save_plots=False,
                         show_validation_accuracy_each_epoch=False):
    adversarial_epochs.restart()
    pgd_robust_net = net
    start_training(pgd_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                   net_name="{} with PGD adversarial training".format(net_name), train_attack=attacks.PGD,
                   attack_training_hps_gen=pgd_training_hps_gen,
                   load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                   save_plots=save_plots,
                   show_validation_accuracy_each_epoch=show_validation_accuracy_each_epoch)


if __name__ == '__main__':
    # initialization
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='MNIST',
                        help='choose one of: [MNIST]')
    parser.add_argument('--seed', type=int, default=0,
                        help='The number of random seed.')
    parser.add_argument('--net', type=str, default='diff_arch',
                        help='To choose what network to train.')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    if args.net == 'robust':
        network_architecture = models.CNN_MNIST_OriginNet
    else:
        network_architecture = models.CNN_MNIST_B

    # load configs from configs.py
    experiment_configs = configs.configs_dict[dataset_name]["configs"]
    experiment_hps_sets = configs.configs_dict[dataset_name]["hps_dict"]
    if args.net != 'robust':
        experiment_hps_sets["PGD_attack"]["steps"] = [40]
        experiment_hps_sets["PGD_train"]["steps"] = [40]
    experiment_results_folder = os.path.join(configs.results_folder, dataset_name)
    experiment_checkpoints_folder = os.path.join(configs.checkpoints_folder, dataset_name)
    logger_path = os.path.join(experiment_results_folder, "log.txt")
    plots_folder = os.path.join(experiment_results_folder, "plots")

    # paths existence validation and initialization
    if not os.path.exists(configs.results_folder):
        os.mkdir(configs.results_folder)
    if not os.path.exists(experiment_results_folder):
        os.mkdir(experiment_results_folder)
    if not os.path.exists(configs.checkpoints_folder):
        os.mkdir(configs.checkpoints_folder)
    if not os.path.exists(experiment_checkpoints_folder):
        os.mkdir(experiment_checkpoints_folder)
    if os.path.exists(plots_folder):
        shutil.rmtree(plots_folder)
        time.sleep(.0001)
    os.mkdir(plots_folder)
    if os.path.exists(logger_path):
        os.remove(logger_path)

    # set logger
    logger.init_log(logger_path)
    logger.log_print("Dataset name: {}".format(dataset_name))
    logger.log_print("checkpoints folder: {}".format(experiment_checkpoints_folder))
    logger.log_print("save checkpoints: {}".format(configs.save_checkpoints))
    logger.log_print("load checkpoints: {}".format(configs.load_checkpoints))
    logger.log_print("results folder: {}".format(experiment_results_folder))
    logger.log_print("show results:  {}".format(configs.show_attacks_plots))
    logger.log_print("save results:  {}".format(configs.save_attacks_plots))

    # Random seed
    seed = args.seed
    paddle.seed(seed)
    logger.log_print("seed: {}".format(seed))

    # 开启0号GPU训练
    use_gpu = True
    device = paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    logger.log_print("execution device: {}".format(device))

    # get datasets
    path_to_save_data = os.path.join(".", "datasets", "mnist")
    _training_dataset = MNIST(mode='train', download=True,
                              transform=paddle.vision.transforms.Compose([paddle.vision.transforms.ToTensor()]))
    _testing_dataset = MNIST(mode='test', download=True,
                             transform=paddle.vision.transforms.Compose([paddle.vision.transforms.ToTensor()]))

    # create hyperparameters generators
    net_training_hps_gen = helper.GridSearch(experiment_hps_sets["nets_training"])
    fgsm_attack_hps_gen = helper.GridSearch(experiment_hps_sets["FGSM_attack"])
    pgd_attack_hps_gen = helper.GridSearch(experiment_hps_sets["PGD_attack"])
    fgsm_training_hps_gen = helper.GridSearch(experiment_hps_sets["FGSM_train"])
    pgd_training_hps_gen = helper.GridSearch(experiment_hps_sets["PGD_train"])

    # loss and general training componenets:
    _loss_fn = experiment_configs["loss_function"]
    training_stop_criteria = experiment_configs["training_stopping_criteria"]
    adv_training_stop_criteria = experiment_configs["adversarial_training_stopping_criteria"]
    epochs = trainer.Epochs(training_stop_criteria)  # epochs obj for not adversarial training
    adv_epochs = trainer.Epochs(adv_training_stop_criteria)  # epochs obj for adversarial training

    # Build robust networks + Compare PGD and FGSM adversarial trainings
    robust_net = network_architecture()
    net_name = robust_net.name
    logger.new_section()
    logger.log_print("Train model on {}".format(net_name))
    build_robust_network(robust_net, _loss_fn, _training_dataset, _testing_dataset, adv_epochs,
                         net_name=net_name,
                         save_checkpoint=configs.save_checkpoints,
                         load_checkpoint=configs.load_checkpoints,
                         show_plots=configs.show_attacks_plots,
                         save_plots=configs.save_attacks_plots,
                         show_validation_accuracy_each_epoch=configs.show_validation_accuracy_each_epoch)
