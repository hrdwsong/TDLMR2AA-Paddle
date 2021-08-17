import os
import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.io import DataLoader
import datasets
import logger
import trainer
import matplotlib
import configs

if not configs.show_attacks_plots:
    matplotlib.use('Agg')


def show_img_lst(imgs, titles=None, x_labels=None, main_title=None, columns=2, plot_img=False, save_img=False,
                 save_path=None):
    """
    Show / save a grid of images (imgs). below each there is a corresponding title / xlabel description - or None
    to ignore this option.
    :param main_title: title of the whole image (above the imgs grid).
    :param columns: number of imgs in each row of the grid. The #rows determined from #columns value.
    :param plot_img: use plt.show?
    :param save_img: use plt.savefig?
    :param save_path: path to save the figure. path to save image. if save_img=True then save_path cannot be None.
    """

    if save_img:
        assert save_path is not None

    # plot images
    img_shape = imgs[0].shape
    fig = plt.figure(figsize=(10, 20))
    rows = np.ceil(len(imgs) / columns)
    for i in range(len(imgs)):
        fig.add_subplot(2 * rows, columns, 2 * i + 1 if i % 2 == 0 else 2 * i)
        img = imgs[i].detach().cpu()
        # fix img
        if img_shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))
        # plot img
        plt.imshow(img)
        if titles is not None:
            plt.xlabel(titles[i])
        if x_labels is not None:
            plt.xlabel(x_labels[i])

    if main_title is not None:
        fig.suptitle(main_title)

    # plot / save
    if plot_img:
        plt.show()
    if save_img:
        plt.savefig(save_path)
    plt.clf()


# Hyperparameter Generation:
class HyperparamsGen:
    """ Abstract class for hyperparameters generation techniques. """

    def __init__(self, hps_dict):
        self.hps_dict = hps_dict
        self.size_num = None

    def next(self):
        """ returns NONE if there are no more hps """
        pass

    def restart(self):
        pass

    def size(self):
        """
            number of possible hyperparams:
            c = 1
            for key in self.hps_dict.keys():
                c *= len(self.hps_dict[key])
            return c
        """
        if self.size_num is None:
            self.size_num = np.prod([len(self.hps_dict[k]) for k in self.hps_dict.keys()])
        return self.size_num


class GridSearch(HyperparamsGen):
    """
        Goes over all possible combinations of hps (hyperparameters).
        Implemented as a generator to save memory - critical when there are many hps.
    """

    def __init__(self, hps_dict):
        super().__init__(hps_dict)
        self.hps_keys = list(hps_dict.keys())
        self.values_size = [len(hps_dict[k]) - 1 for k in self.hps_keys]
        self.indices = [0] * len(self.hps_keys)

    def next(self):
        """ returns NONE if there are no more hps"""
        if self.indices[0] > self.values_size[0]:
            return None

        # construct HP:
        hp = {}
        for idx, val_idx in enumerate(self.indices):
            key = self.hps_keys[idx]
            hp[key] = self.hps_dict[key][val_idx]

        # next hp indices
        i = len(self.indices) - 1
        while i >= 0 and self.indices[i] == self.values_size[i]:
            self.indices[i] = 0
            i -= 1
        self.indices[max(0, i)] += 1 if i >= 0 else self.values_size[0] + 1

        return hp

    def restart(self):
        """ restarts generator to re-use it in hps search """
        self.indices = [0] * len(self.hps_keys)


def concat_hps_gens(hps1: HyperparamsGen, hps2: HyperparamsGen):
    concat_hps_dict = {}
    concat_hps_dict.update(hps1.hps_dict)
    concat_hps_dict.update(hps2.hps_dict)

    return hps1.__class__(concat_hps_dict)


def hps_search(hp_gen: HyperparamsGen, func, *params):
    pass


def measure_resistance_on_test(robust_net, blackbox_net, loss_fn, test_dataset, to_attacks, num_restarts=1, device=None, plots_title="", plot_results=False,
                               save_figs=False, figs_path=None):
    """
    measure the trained net resistance to the specified attacks (to_attacks) on test dataset. has option
    to save / plot the successful attacks.
    """

    results = {}
    test_dataloader = DataLoader(test_dataset, batch_size=100)
    original_acc = trainer.measure_classification_accuracy(robust_net, test_dataloader, device=device)
    for attack_class, attack_hp in to_attacks:
        attack = attack_class(blackbox_net, loss_fn, attack_hp, rand=True)
        title = "{}_{}".format(attack.name, plots_title)
        test_acc = attack.test_attack(robust_net,
                                      test_dataloader,
                                      num_restarts=num_restarts,
                                      main_title=title,
                                      plot_results=plot_results,
                                      save_results_figs=save_figs,
                                      fig_path=os.path.join(figs_path, "{}.png".format(title)),
                                      device=device)
        results["%{}".format(attack.name)] = test_acc

    results["test_acc"] = original_acc
    return results


def full_train_of_nn_with_hps(net, loss_fn, train_dataset, hps_gen, epochs, device=None, train_attack=None,
                              full_train=False, show_validation=False, add_natural_examples=False):
    """
    Here we do hyperparameter search to find best training hyperparameter.
    Apply cross validation training and measuring on the hyperparameters and choose the one with best validation measurements.
    Enables to train as adversarial training by specifying train_attack.

    :param net: net to train (its parameters will be initialized)
    :param loss_fn: loss function.
    :param train_dataset: dataset to train on.
    :param hps_gen: hyperparameter generator - HyperparamsGen type. we iterate on this object.
    :param epochs: Epochs object to manage stopping methodology.
    :param device: the device to execute on.
    :param train_attack: in case we do an adversarial training. Its hp (parameters) should be given also (tuple).
    :param full_train: train the net on all dataset on the selected hyperparameter.
    :param show_validation: show also validation measurements on training log.
    :param add_natural_examples: add natural training examples. relevant only for adversarial training (i.e.
           train_attack is not None).
    :return: net, net_best_hp, net_best_acc. net is trained on full train dataset (not splitted)
    """
    early_stop = isinstance(epochs, trainer.EarlyStopping)
    if early_stop:
        full_train = False  # equivalent

    hps_gen.restart()
    best_net_state_dict, net_best_hp, net_best_acc = None, None, 0
    if hps_gen.size() > 1 or (hps_gen.size() == 1 and early_stop):
        while True:
            hp = hps_gen.next()
            if hp is None:
                break
            logger.log_print("\nTesting: {}".format(str(hp)))

            # restart previous execution
            epochs.restart()

            # set train and val dataloaders, optimizer
            train_dl, val_dl = datasets.get_train_val_dls(train_dataset, hp["batch_size"])
            if hp["lr_scheduler_gamma"] is not None:
                lr_scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=hp["lr"],
                                                                    gamma=hp["lr_scheduler_gamma"])
                nn_optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=lr_scheduler)
                epochs.set_lr_scheduler(lr_scheduler)
            else:
                nn_optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=hp["lr"])

            # train network
            _val_dl = None
            if show_validation or early_stop:
                _val_dl = val_dl
            # define attack using hp
            attack_obj = None
            if train_attack is not None:
                attack_obj = train_attack(net, loss_fn, hp)

            trainer.train_nn(net, nn_optimizer, loss_fn, train_dl, epochs, device=device, attack=attack_obj,
                             val_dl=_val_dl, add_natural_examples=add_natural_examples)

            # measure on validation set
            net_acc = trainer.measure_classification_accuracy(net, val_dl, device=device)
            logger.log_print("hp {} with val acc: {}".format(str(hp), net_acc))
            if net_acc >= net_best_acc:
                net_best_acc = net_acc
                net_best_hp = hp
                best_net_state_dict = net.state_dict()
    else:
        net_best_hp = hps_gen.next()

    if full_train or (hps_gen.size() == 1 and not early_stop):
        logger.log_print("\nFull Train(training on all training dataset) with selected hp: {}".format(str(net_best_hp)))
        epochs.restart()
        full_train_dl = DataLoader(train_dataset, batch_size=net_best_hp["batch_size"], shuffle=True, use_shared_memory=True)
        nn_optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=net_best_hp["lr"])
        attack_obj = None
        if train_attack is not None:
            attack_obj = train_attack(net, loss_fn, net_best_hp)

        trainer.train_nn(net, nn_optimizer, loss_fn, full_train_dl, epochs, device=device, attack=attack_obj,
                         add_natural_examples=add_natural_examples)
        best_net_state_dict = net.state_dict()

    return best_net_state_dict, net_best_hp


def full_attack_of_trained_nn_with_hps(net, loss_fn, train_dataset, hps_gen, selected_nn_hp, attack_method, device=None,
                                       plots_title="", plot_results=False, save_figs=False, figs_path=None):
    """
    hyperparameter search in order to find the hp with highest attack score (i.e. prob to successfully attack).
    :param net: net to train (its parameters will be initialized)
    :param loss_fn: loss function.
    :param train_dataset: dataset to train on.
    :param hps_gen: hyperparameter generator - HyperparamsGen type. we iterate on this object.
    :param selected_nn_hp: selected hyperparameter from training hyperparameter search section.
    :param attack_method: of Attack type. we choose the hp that maximizes this attack score.
    :param device: device to execute on.
    :param plots_title: title of the plots
    :param plot_results: use plt.show()?
    :param save_figs: use plt.savefig(...)?
    if both plot_results and save_figs we don't construct the grid view of the successful adversarial examples.
    :param figs_path: where to save the figure.
    :return: best_hp, best_score (approximately prob to successfully attack)
    """
    hps_gen.restart()
    train_dl, val_dl = datasets.get_train_val_dls(train_dataset, selected_nn_hp["batch_size"])

    best_hp, lowest_acc = None, 1.0
    while True:
        hp = hps_gen.next()
        if hp is None:
            break

        attack = attack_method(net, loss_fn, hp, rand=True)
        title = "resistance {}: {}".format(attack.name, plots_title)
        acc_on_attack = attack.test_attack(
            net,
            val_dl,
            main_title=title,
            plot_results=plot_results,
            save_results_figs=save_figs,
            fig_path=os.path.join(figs_path, title),
            device=device
        )
        if acc_on_attack <= lowest_acc:
            lowest_acc = acc_on_attack
            best_hp = hp

        logger.log_print("%accuracy on attack: {} with hp: {}".format(acc_on_attack, str(hp)))

    return best_hp, lowest_acc

