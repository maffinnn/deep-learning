import numpy as np

from layers import FCLayer

from dataloader import build_dataloader
from network import Network
from loss import SoftmaxCrossEntropyLoss
from optimizer import SGD


from visualize import plot_loss_and_acc

class Solver(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # build dataloader
        train_loader, val_loader, test_loader = self.build_loader(cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # build model
        self.model = self.build_model(cfg)

        # build optimizer
        self.optimizer = self.build_optimizer(self.model, cfg)

        # build evaluation criterion
        self.criterion = SoftmaxCrossEntropyLoss()

    @staticmethod
    def build_loader(cfg):
        train_loader = build_dataloader(
            cfg['data_root'], cfg['max_epoch'], cfg['batch_size'], shuffle=True, mode='train')

        val_loader = build_dataloader(
            cfg['data_root'], 1, cfg['batch_size'], shuffle=False, mode='val')

        test_loader = build_dataloader(
            cfg['data_root'], 1, cfg['batch_size'], shuffle=False, mode='test')

        return train_loader, val_loader, test_loader

    @staticmethod
    def build_model(cfg):
        model = Network()
        model.add(FCLayer(784, 10)) 
        return model

    @staticmethod
    def build_optimizer(model, cfg):
        return SGD(model, cfg['learning_rate'], cfg['momentum'])

    def train(self):
        # print("training......")
        max_epoch = self.cfg['max_epoch']

        epoch_train_loss, epoch_train_acc = [], []
        for epoch in range(max_epoch):

            iteration_train_loss, iteration_train_acc = [], []
            for iteration, (images, labels) in enumerate(self.train_loader):
                # print("current iteration:", iteration)
                # print("images.type:", type(images))
                # print("images.shape:", images.shape)
                # print("labels.shape:", labels.shape)
                # forward pass
                logits = self.model.forward(images)
                loss, acc = self.criterion.forward(logits, labels)

                # backward_pass
                delta = self.criterion.backward()
                self.model.backward(delta)

                # update the model weights
                self.optimizer.step()

                # restore loss and accuracy
                iteration_train_loss.append(loss)
                iteration_train_acc.append(acc)

                # display iteration training info
                if iteration % self.cfg['display_freq'] == 0:
                    print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                        epoch, max_epoch, iteration, len(self.train_loader), loss, acc))

            avg_train_loss, avg_train_acc = np.mean(iteration_train_loss), np.mean(iteration_train_acc)
            epoch_train_loss.append(avg_train_loss)
            epoch_train_acc.append(avg_train_acc)

            # validate
            avg_val_loss, avg_val_acc = self.validate()

            # display epoch training info
            print('\nEpoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
                epoch, avg_train_loss, avg_train_acc))

            # display epoch valiation info
            print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}\n'.format(
                epoch, avg_val_loss, avg_val_acc))

        return epoch_train_loss, epoch_train_acc

    def validate(self):
        logits_set, labels_set = [], []
        for images, labels in self.val_loader:
            logits = self.model.forward(images)
            logits_set.append(logits)
            labels_set.append(labels)

        logits = np.concatenate(logits_set)
        labels = np.concatenate(labels_set)
        loss, acc = self.criterion.forward(logits, labels)
        return loss, acc

    def test(self):
        logits_set, labels_set = [], []
        for images, labels in self.test_loader:
            logits = self.model.forward(images)
            logits_set.append(logits)
            labels_set.append(labels)

        logits = np.concatenate(logits_set)
        labels = np.concatenate(labels_set)
        loss, acc = self.criterion.forward(logits, labels)
        return loss, acc


if __name__ == '__main__':
    # You can modify the hyerparameters by yourself.
    relu_cfg = {
        'data_root': 'data',
        'max_epoch': 10,
        'batch_size': 100,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'display_freq': 50,
        'activation_function': 'relu',
    }

    runner = Solver(relu_cfg)
    relu_loss, relu_acc = runner.train()

    test_loss, test_acc = runner.test()
    print('Final test accuracy {:.4f}\n'.format(test_acc))

    # You can modify the hyerparameters by yourself.
    sigmoid_cfg = {
        'data_root': 'data',
        'max_epoch': 10,
        'batch_size': 100,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'display_freq': 50,
        'activation_function': 'sigmoid',
    }

    runner = Solver(sigmoid_cfg)
    sigmoid_loss, sigmoid_acc = runner.train()

    test_loss, test_acc = runner.test()
    print('Final test accuracy {:.4f}\n'.format(test_acc))

    plot_loss_and_acc({
        "relu": [relu_loss, relu_acc],
        "sigmoid": [sigmoid_loss, sigmoid_acc],
    })
