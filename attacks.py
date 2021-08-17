import paddle


class Attack:
    def __init__(self, net, loss_fn):
        """
        :param net: the network to attack
        :param loss_fn: the attack is with respect to this loss
        """
        self.net = net
        self.loss_fn = loss_fn
        self.name = None
        self.device = None

    def perturb(self, X, y, device=None):
        """ generates adversarial examples to inputs (X, y) """
        pass

    def test_attack(self, robust_net, dataloader, attacted_net=None, num_restarts=1, plot_results=True, save_results_figs=True, fig_path=None, main_title="",
                    device=None):
        """
        the attack score of attack method A on network <net> is E[A(x) != y] over distribution D when A(x) is the
        constructed adversarial example of attack A on x. We are going to estimate it using samples from test_dataset.

        :param plot_results: plot original vs adv imgs with additional information
        :return: the accuracy on constructed adversarial examples (i.e. 1 - attack score).
        """
        # calculate attack score
        self.device = device
        if attacted_net is None:
            attacted_net = self.net
        num_successful_attacks = 0
        successful_attacks_details = []
        for batch_num, (xs, ys) in enumerate(dataloader):
            ys = ys[:, 0]
            # print('batch_num: {}'.format(batch_num))
            for _ in range(num_restarts):
                # calculate %successful attacks on xs, ys
                constructed_examples = self.perturb(xs, ys, device=device)
                adv_y_preds = robust_net(constructed_examples)
                hard_adv_y_preds = paddle.argmax(adv_y_preds, axis=1)
                batch_successful_attack = paddle.not_equal(hard_adv_y_preds, ys)
                num_successful_attacks += batch_successful_attack.numpy().sum().item()

        dataloader_size = len(dataloader) * dataloader.batch_size
        attack_score = num_successful_attacks / (dataloader_size * num_restarts)
        acc_on_constructed_examples = 1.0 - attack_score

        return acc_on_constructed_examples


class FGSM(Attack):
    def __init__(self, net, loss_fn, hp, rand=False):
        super().__init__(net, loss_fn)
        self.epsilon = hp["epsilon"]
        self.name = "fgsm"

    def perturb(self, X, y, device=None):
        """ generates adversarial examples to given data points and labels (X, y) based on FGSM approach. """

        X.stop_gradient = False
        y_pred = self.net(X)

        self.net.clear_gradients()
        loss = self.loss_fn(y_pred, y)
        loss.backward()

        adv_X = X + self.epsilon * X.grad.sign()
        adv_X = paddle.clip(adv_X, 0, 1)

        return adv_X


class PGD(Attack):
    def __init__(self, net, loss_fn, hp, rand=False):
        super().__init__(net, loss_fn)
        self.steps = hp["steps"]
        self.alpha = hp["alpha"]
        self.epsilon = hp["epsilon"]
        self.name = "pgd"
        self.rand = rand

    def perturb(self, X, y, device=None):
        """ generates adversarial examples to given data points and labels (X, y) based on PGD approach. """
        if self.rand:
            original_X = X
            X = X + paddle.uniform(X.shape, min=-self.epsilon, max=self.epsilon)
            X = paddle.clip(X, 0, 1)  # ensure valid pixel range
        else:
            original_X = X

        for i in range(self.steps):
            X.stop_gradient = False
            outputs = self.net(X)
            _loss = self.loss_fn(outputs, y)
            _loss.backward()

            X = X + self.alpha * X.grad.sign()
            diff = paddle.clip(X - original_X, min=-self.epsilon, max=self.epsilon)  # gradient projection
            X = paddle.clip(original_X + diff, min=0.0, max=1.0).detach()  # to stay in image range [0,1]

        return X

