from typing import *
import numpy as np


def get_top_values(lst: Iterable, n: int , labels: Dict[int, Any]):
    #Give a list of values, find the indices with the highest n values
    #Return the labels for each of the indices
    return [labels[i] for i in np.argsort(lst)[::-1][:n]]

def get_bottom_values(lst: Iterable, n: int , labels: Dict[int, Any]):
    #Give a list of values, find the indices with the lowest n values
    #Return the labels for each of the indices
    return [labels[i] for i in np.argsort(lst)[:n]]