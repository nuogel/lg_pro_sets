"""Calculate the F1 score."""
from lgdet.registry import SCORES
import torch


@SCORES.registry()
class Score:
    """Calculate the F1 score."""

    def __init__(self, cfg):
        """Init the parameters."""
        ...

    def init_parameters(self):
        self.score = 0.
        self.batches = 0

    def cal_score(self, pre_labels, test_data=None, from_net=True):
        gt_labels = test_data[1]
        self.score += (torch.sum((torch.argmax(torch.softmax(pre_labels, -1)) == gt_labels)) / pre_labels.size(0)).item()
        self.batches += 1

    def score_out(self):
        score_out = self.score / self.batches
        print('test class score:', score_out)
        return score_out, None
