"""Saving the parameters while training, and drawing."""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from cfg.yml_parse import parse_yaml


class TrainParame:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.file = cfg.PATH.PARAMETER_PATH

    def save_parameters(self, epoch,
                        learning_rate=None, batch_average_loss=None,
                        f1_score=None, precision=None, recall=None):
        # pylint: disable=too-many-arguments
        """Save the parameters."""
        # check self.file
        if not os.path.isfile(self.file):
            dict_ = {'epoch': 0,
                     'learning_rate': [self.cfg.TRAIN.LR_START],
                     'batch_average_loss': [],
                     'f1_score': [],
                     'precision': [],
                     'recall': []}
            if not os.path.isdir(os.path.dirname(self.file)):
                os.mkdir(os.path.dirname(self.file))
            torch.save(dict_, self.file)
        dict_loaded = torch.load(self.file)
        dict_loaded['epoch'] = epoch
        if learning_rate is not None:
            dict_loaded['learning_rate'] = dict_loaded['learning_rate'][0:epoch + 1]
            dict_loaded['learning_rate'].append(learning_rate)
        if batch_average_loss is not None:
            dict_loaded['batch_average_loss'] = dict_loaded['batch_average_loss'][0:epoch + 1]
            dict_loaded['batch_average_loss'].append(batch_average_loss)
        if f1_score is not None:
            dict_loaded['f1_score'] = dict_loaded['f1_score'][0:epoch + 1]
            dict_loaded['f1_score'].append(f1_score)
        if precision is not None:
            dict_loaded['precision'] = dict_loaded['precision'][0:epoch + 1]
            dict_loaded['precision'].append(precision)
        if recall is not None:
            dict_loaded['recall'] = dict_loaded['recall'][0:epoch + 1]
            dict_loaded['recall'].append(recall)

        torch.save(dict_loaded, self.file)

    def clear_parameters(self):
        """Init the parameters."""
        if os.path.isfile(self.file):
            os.remove(self.file)

    def show_parameters(self, start_epoch=0, end_epoch=None):
        """Show the parameters."""

        def _transpose_list(list_in):
            list_out = np.array(list_in).transpose()
            return list_out

        if os.path.isfile(self.file):
            print(self.file)
            dict_loaded = torch.load(self.file)

            learning_rate = dict_loaded['learning_rate']
            batch_average_loss = dict_loaded['batch_average_loss']
            f1_score = dict_loaded['f1_score']
            precision = dict_loaded['precision']
            recall = dict_loaded['recall']
            # print(dict_loaded)

            f1_score = _transpose_list(f1_score)
            precision = _transpose_list(precision)
            recall = _transpose_list(recall)

            self._draw_img([learning_rate, batch_average_loss, f1_score, precision, recall], start_epoch, end_epoch)
        else:
            print('no parameter.file is found.')

    def _draw_img(self, datas, start_epoch=0, end_epoch=None):
        """Show the parameters."""

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'brown', 'darkred', 'linen']
        linestyles = ['-', '-.', '--', ':', '-', '-.', '--', ':', '-', '-.', '--', ':', ]

        labels = ['+', 'o', '*', '.', 'x', 'p', 'H', 'h', '^', '>', '<', '1', '2', '3', '4']
        line_illustration = ['learning_rate', 'batch_average_loss', 'f1_score', 'precision', 'recall']

        for i, data in enumerate(datas):
            if data is []:  # if there is no data then continue
                continue
            end_epoch = len(data) - 1 if not end_epoch else end_epoch
            data_x1 = np.array(range(len(data)))[start_epoch:end_epoch + 1]
            data_y1 = np.array(data, dtype=np.float32)[start_epoch:end_epoch + 1]
            plt.figure(num=1, figsize=(20, 10))
            plt.subplot(321 + i)
            plt.xlabel('Epoch')
            plt.ylabel(line_illustration[i])
            if len(data_y1) == 0:  # if there is no data in it ,then do nothing.
                continue
            if data_y1.shape.__len__() == 1:
                plt.plot(data_x1, data_y1, color=colors[i], linewidth=3,
                         linestyle=linestyles[i], label=labels[i])
                # print(data_x1[-1], data_y1[-1])
                plt.text(x=data_x1[-1], y=data_y1[-1], s='(' + str(data_x1[-1]) + ',' + str(data_y1[-1]) + ')')
            else:
                data_x = range(data_y1.shape[1])[start_epoch:end_epoch + 1]
                for j, data_y in enumerate(data_y1):
                    plt.plot(data_x, data_y[data_x], color=(colors[j]),
                             linewidth=3,
                             linestyle=linestyles[j],
                             label=self.cfg.TRAIN.CLASSES[j] + str('(%0.3f)' % data_y[data_x[-1]]))
                    plt.legend(loc='lower right')

        plt.show()


if __name__ == "__main__":
    """Test show_parameters."""
    cfg = parse_yaml('../cfg/ASR.yml')
    para = TrainParame(cfg)
    para.show_parameters(1,)
