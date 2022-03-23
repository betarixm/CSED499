from abc import ABC, abstractmethod
from typing import NewType

import tensorflow as tf

TestSet = NewType("TestSet", tf.data.Dataset)
TrainSet = NewType("TrainSet", tf.data.Dataset)


class Dataset(ABC):
    @staticmethod
    @abstractmethod
    def dataset() -> (TestSet, TrainSet):
        pass

    def __new__(cls, *args, **kwargs):
        return cls.dataset()
