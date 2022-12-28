import numpy as np


class One_Hot_Encoder:

    @staticmethod
    def fit_transform(arr, unique_arr):
        encoder_dict = {}
        j = 0
        for i in unique_arr:
            encoder_dict[j] = i
            j += 1
        res = np.zeros((arr.shape[0], len(unique_arr)))
        for i in range(arr.shape[0]):
            res[i][encoder_dict[arr[i]] - 1] = 1

        return res
