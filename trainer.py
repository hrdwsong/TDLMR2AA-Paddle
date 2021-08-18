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

import time
import paddle
from paddle.io import DataLoader
import logger


class StoppingCriteria:
    def __init__(self):
        self.epoch_num = 0
        self.epochs_summaries = list()

    def update(self, epochs_summaries):
        """UPDATE AFTER EVERY EPOCH"""
        self.epoch_num += 1
        self.epochs_summaries = epochs_summaries  # by pointer - faster then append

    def stop(self):
        pass

    def restart(self):
        pass


class EarlyStopping(StoppingCriteria):
    """ To use this epochs method a val_dl must be specified """

    def __init__(self, max_epochs_num):
        """
        :param max_epochs_num: number of epochs to stop after
        """
        super().__init__()
        self.max_epochs_num = max_epochs_num

    def stop(self):
        if self.epochs_summaries[-1]["val_acc"] is not None and self.epoch_num > 1:
            improved_con = self.epochs_summaries[-2]["val_acc"] < self.epochs_summaries[-1]["val_acc"]
        else:
            improved_con = True
        max_epochs_con = True if self.epoch_num >= self.max_epochs_num else False
        return improved_con and max_epochs_con

    def restart(self):
        self.epoch_num = 0


class ConstantStopping(StoppingCriteria):
    def __init__(self, max_epochs_num):
        """
        :param max_epochs_num: number of epochs to stop after
        """
        super().__init__()
        self.max_epochs_num = max_epochs_num

    def stop(self):
        return True if self.epoch_num >= self.max_epochs_num else False

    def restart(self):
        self.epoch_num = 0


class TimerStopping(StoppingCriteria):
    def __init__(self, max_time):
        """
        :param max_time: number of seconds to stop after. It will be with a delay of at most epoch training time.
        """
        super().__init__()
        self.max_time = max_time
        self.start_time = time.time()

    def stop(self):
        curr_time = time.time()
        return True if curr_time - self.start_time > self.max_time else False

    def restart(self):
        self.start_time = time.time()


class Epochs:
    def __init__(self, stopping_criteria: StoppingCriteria):
        self.epoch_number = 0
        self.epochs_summaries = list()
        self.stopping_criteria = stopping_criteria
        self.latest_net_params = [None, None]
        self.lr_scheduler = None

    def update(self, epoch_summary, state_dict):
        self.epoch_number += 1
        self.epochs_summaries.append(epoch_summary)
        self.stopping_criteria.update(self.epochs_summaries)
        self.latest_net_params[0] = self.latest_net_params[1]
        self.latest_net_params[1] = state_dict

    def stop(self):
        return self.stopping_criteria.stop()

    def fix_weights(self, net):
        if isinstance(self.stopping_criteria, EarlyStopping):  # 这行没有执行
            net.set_state_dict(self.latest_net_params[0])  # 此处应该是1吧？？？

    def print_last_epoch_summary(self):
        summary = self.epochs_summaries[-1]
        msg = "Epoch {}. ".format(len(self.epochs_summaries))
        if summary["adv_acc"] != -1:
            msg += "Train adversarial accuracy: {val_acc:2f}.".format(val_acc=summary["adv_acc"].item())
        if summary["val_acc"] is not None:
            msg += "Validation accuracy: {val_acc:2f}.".format(val_acc=summary["val_acc"].item())
        msg += "Train accuracy: {train_acc:.2f},  Train loss:  {train_loss:.2f}\n".format(train_acc=summary["acc"].item(),
                                                                                          train_loss=summary["loss"].item())
        logger.log_print(msg)

    def restart(self):
        self.epoch_number = 0
        self.epochs_summaries = list()
        self.stopping_criteria.restart()
        self.latest_net_params = [None, None]
        self.lr_scheduler = None

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def adjust_lr(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


def train_nn(net, optimizer, loss_fn, train_dl, epochs: Epochs, attack=None, device=None, val_dl=None,
             add_natural_examples=False):
    """
    :param net: the network to train :)
    :param optimizer: torch.optim optimizer (e.g. SGD or Adam).
    :param loss_fn: we train with respect to this loss.
    :param train_dl: train data loader. We use that to iterate over the train data.
    :param val_dl: validation data loader. We use that to iterate over the validation data to measure performance.
                   None for ignore. If epochs uses Early Stopping then val_dl cannot be None.
    :param epochs: number of epochs to train or early stopping.
    :param attack: the attack we want to defense from - only PGD and FGSM are implemented. None for natural training
                   (i.e. with no resistance to any specific attack).
    :param device: use cuda or cpu
    """

    while not epochs.stop():
        batch_information_mat = []

        for batch_num, (batch_data, batch_labels) in enumerate(train_dl):
            # train on natural batch
            batch_labels = batch_labels[:, 0]
            if attack is None or add_natural_examples:  # True
                batch_preds = net(batch_data)
                _loss = loss_fn(batch_preds, batch_labels)
                optimizer.clear_grad()
                _loss.backward()
                optimizer.step()

            # train on constructed adversarial examples (Adversarial Training Mode)
            if attack is not None:
                if not add_natural_examples:
                    with paddle.no_grad():
                        batch_preds = net(batch_data)  # evaluate on natural examples

                adversarial_batch_data = attack.perturb(batch_data, batch_labels, device=device)
                adversarial_batch_preds = net(adversarial_batch_data)
                _loss = loss_fn(adversarial_batch_preds, batch_labels)
                optimizer.clear_grad()
                _loss.backward()
                optimizer.step()

            # calculate batch measurements
            hard_batch_preds = paddle.argmax(batch_preds, axis=1)
            batch_num_currect = paddle.equal(hard_batch_preds, batch_labels).numpy().sum().item()
            batch_acc = batch_num_currect / len(batch_labels)
            adv_batch_acc = -1
            if attack is not None:
                hard_adv_batch_preds = paddle.argmax(adversarial_batch_preds, axis=1)
                adv_batch_num_currect = paddle.equal(hard_adv_batch_preds, batch_labels).numpy().sum().item()
                adv_batch_acc = adv_batch_num_currect / len(adversarial_batch_preds)

            batch_information_mat.append([_loss.item(), batch_acc, adv_batch_acc])

        # summarize epoch (should be a part of the log):
        batch_information_mat = paddle.to_tensor(batch_information_mat)
        emp_loss, emp_acc, adv_acc = paddle.sum(batch_information_mat.transpose([1, 0]), axis=1) / len(batch_information_mat)  # 可能有问题，调试看下batch_information_mat的shape
        if val_dl is not None:
            val_acc = measure_classification_accuracy(net, val_dl, device=device)
        else:
            val_acc = None
        curr_epoch_summary = {"acc": emp_acc, "loss": emp_loss, "val_acc": val_acc, "adv_acc": adv_acc}
        epochs.update(curr_epoch_summary, net.state_dict())
        epochs.print_last_epoch_summary()
        epochs.adjust_lr()

    epochs.fix_weights(net)


def measure_classification_accuracy(trained_net, dataloader: DataLoader, device=None):
    """
    couldn't load the whole dataset into GPU memory so the function calculates it in batches.
    :param dataloader: dataloader with data to measure on.
    :param trained_net: trained network to measure on dl.
    :return: accuracy (of trained_net on dl)
    """
    correct_classified = 0
    for xs, ys in dataloader:
        # xs, ys = xs.to(device), ys.to(device)
        ys = ys[:, 0]
        ys_pred = trained_net(xs)
        hard_ys_pred = paddle.argmax(ys_pred, axis=1)
        correct_classified += paddle.equal(ys, hard_ys_pred).numpy().sum().item()
    dataloader_size = len(dataloader) * dataloader.batch_size
    return correct_classified / dataloader_size
