import pickle
import numpy as np

# ファイルを読み込む
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        fo.close()
    return data_dict

class CIFARDATA:
    def __init__(self) -> None:
        data_names = [
            '.\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_1',
            '.\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_2',
            '.\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_3',
            '.\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_4',
            '.\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_5'
        ]

        self.labels = []
        self.dataset = []
        for data_name in data_names:
            read_result = unpickle(data_name)
            self.labels.append(read_result[b'labels'])
            self.dataset.append(read_result[b'data'])

        self.y = np.concatenate(self.labels)
        self.x = np.concatenate(self.dataset)
        self.y_ = unpickle(
            '.\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\test_batch')[b'labels']
        self.x_ = unpickle(
            '.\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\test_batch')[b'data']

    def GetDataSet(self):
        return self.dataset