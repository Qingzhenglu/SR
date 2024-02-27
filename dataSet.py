import torch
from matplotlib import pyplot as plt
from torchaudio.datasets import SPEECHCOMMANDS
import os

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
          'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
          'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
          'up', 'visual', 'wow', 'yes', 'zero']


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")
# validation_set = SubsetSC("validation")

# waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


def label_to_index(word):
    return labels.index(word)


def index_to_label(index):
    return labels[index]


# word_test = "backward"
# # index_test = 15
# wt_toIndex = label_to_index(word_test)
# it_toWord = index_to_label(wt_toIndex)
# print(word_test, "->", wt_toIndex, "->", it_toWord)
