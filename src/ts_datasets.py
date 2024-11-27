"""
get data sets
"""
import os
import math
import glob
import logging
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import numpy.typing as npt

import torch
from torch.utils.data import DataLoader

# from momentfm.utils.data import load_from_tsfile, convert_tsf_to_dataframe
from models.moment_prompt.utils.data import load_from_tsfile, convert_tsf_to_dataframe

import sys
sys.path.append('../')
sys.path.append('../data/mimic3_benchmarks')

# from mimic3_benchmarks.mimic3benchmark.readers import InHospitalMortalityReader
# from mimic3_benchmarks.mimic3benchmark.readers import PhenotypingReader
# from mimic3_benchmarks.mimic3models.preprocessing import Discretizer, Normalizer
# from mimic3_benchmarks.mimic3models.in_hospital_mortality import utils

from typing import Optional


# UCR: https://timeseriesclassification.com/dataset.php



DATASETS_WITHOUT_NORMALIZATION = [
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "BME",
    "Chinatown",
    "Crop",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "MelbournePedestrian",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "PowerCons",
    "Rock",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "SmoothSubspace",
    "UMD",
]


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, batch_size: int,
                 filename: str,
                 data_split: str = "train",
                 ):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """
        self.seq_len = 512
        self.batch_size = batch_size

        self.data_split = data_split  # 'train' or 'test'
        self.filename =  filename
        if data_split == "train" or data_split == "val":
            self.filename += "_TRAIN.ts"
        elif data_split == "test":
            self.filename += "_TEST.ts"

        self.train_ratio = 0.6
        self.val_ratio = 0.1

        self.dataset_name = self.filename.split("/")[-1].split("_")[0]
        # get labels
        _, self.train_labels = load_from_tsfile(self.filename.replace(self.data_split.upper(), "TRAIN"))
        # get data
        self._read_data(self.filename)


    def _transform_labels(self, labels: np.ndarray):
        lbls = np.unique(self.train_labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(lbls):
            transform[l] = i

        labels = np.vectorize(transform.get)(labels)
        return labels


    def _get_borders(self):
        train_end = 0
        val_start = 0
        val_end = 0
        if self.data_split in ["train", "val"]:
            train_ratio, val_ratio = (
                self.train_ratio / (self.train_ratio + self.val_ratio),
                self.val_ratio / (self.train_ratio + self.val_ratio),
            )
            train_end = int(train_ratio * self.n_timeseries)

        return slice(0, train_end), slice(train_end, None), slice(0, None)


    def _check_if_equal_length(self):
        if isinstance(self.data, list):
            n_timeseries = len(self.data)

            self.n_channels = self.data[0].shape[0]
            # Assume all time-series have the same number of channels
            # Then we have time-series of unequal lengths
            max_len = max([ts.shape[-1] for ts in self.data])
            for i, ts in enumerate(self.data):
                self.data[i] = interpolate_timeseries(
                    timeseries=ts, interp_length=max_len, channel_first=True
                )
            self.data = np.asarray(self.data)
            logging.info(
                f"Time-series have unequal lengths. Reshaping to {self.data.shape}"
            )

    def _check_and_remove_nans(self):
        if np.isnan(self.data).any():
            logging.info("NaNs detected. Imputing values...")
            self.data = interpolate_timeseries(
                timeseries=self.data, interp_length=self.data.shape[-1], channel_first=True
            )
            self.data = np.nan_to_num(self.data)


    def _read_data(self, filename: str):
        self.scaler = StandardScaler()

        self.data, self.labels = load_from_tsfile(filename)
        self.labels = self._transform_labels(self.labels)

        self.num_classes = len(np.unique(self.labels))

        self.original_length = None

        self._check_if_equal_length()
        self._check_and_remove_nans()


        if self.data.ndim == 3:
            self.n_timeseries, self.n_channels, self.length_timeseries = self.data.shape
        else:
            self.n_timeseries, self.length_timeseries = self.data.shape
            self.n_channels = 1

        if self.dataset_name not in DATASETS_WITHOUT_NORMALIZATION:
            length_timeseries = self.data.shape[0]
            self.data = self.data.reshape(-1, self.data.shape[-1])
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)
            self.data = self.data.reshape(length_timeseries, -1, self.data.shape[-1])

        idx = np.random.permutation(len(self.data))
        if isinstance(self.data, np.ndarray):
            self.data = self.data[idx]
        else:
            self.data = [self.data[i] for i in idx]
        self.labels = self.labels[idx]


        data_splits = self._get_borders()

        if self.data_split == "train":
            self.data = self.data[data_splits[0],]
            self.labels = self.labels[data_splits[0]]
        elif self.data_split == "val":
            self.data = self.data[data_splits[1],]
            self.labels = self.labels[data_splits[1]]
        elif self.data_split == "test":
            self.data = self.data[data_splits[2],]
            self.labels = self.labels[data_splits[2]]

        self.n_timeseries = self.data.shape[0]
        self.data = self.data.transpose(0, 2, 1)



    def __getitem__(self, index: int):

        timeseries = self.data[index]
        timeseries_len = timeseries.shape[0]
        labels = self.labels[index,].astype(int)

        if timeseries_len <= self.seq_len:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len,
                direction="backward",
                sampling_type="pad",
                mode="constant",
            )

        elif timeseries_len > self.seq_len:
            timeseries, input_mask = downsample_timeseries(
                timeseries, self.seq_len, sampling_type="interpolate"
            )

        return timeseries.T, input_mask, labels


    def __len__(self):
        return self.n_timeseries // self.batch_size * self.batch_size





class InformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        batch_size: int,
        filename: str,
        forecast_horizon: int = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        """
        self.batch_size = batch_size

        self.filename = filename
        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name

        # get data
        self._read_data(filename)


    def _get_borders(self, filename: str):
        self.train_ratio = 0.6
        self.val_ratio = 0.1
        self.test_ratio = 0.3

        if "ETTm" in filename:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4

        elif "ETTh" in filename:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        else:
            n_train = int(self.train_ratio * self.length_timeseries_original)
            n_test = int(self.test_ratio * self.length_timeseries_original)
            n_val = self.length_timeseries_original - n_train - n_test

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        val = slice(train_end - self.seq_len, val_end)
        test = slice(test_start, test_end)

        assert train_end > 0 and val_end > 0 and test_start > 0 and test_end > 0 and train_end - self.seq_len > 0

        return train, val, test


    def _read_data(self, filename: str):
        self.scaler = StandardScaler()
        df = pd.read_csv(filename)
        
        # df = df[['OT']]
        
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders(filename)

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "val":
            self.data = df[data_splits[1], :]
        elif self.data_split == "test":
            self.data = df[data_splits[2], :]

        self.length_timeseries = self.data.shape[0]


    def __getitem__(self, index: int):

        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        assert(seq_start >=0 and seq_end >= 0 and pred_end >= 0)
        timeseries = self.data[seq_start:seq_end, :].T
        forecast = self.data[seq_end:pred_end, :].T

        return timeseries, forecast, input_mask


    def __len__(self):
        return (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.data_stride_len + 1




def upsample_timeseries(
    timeseries: npt.NDArray,
    seq_len: int,
    sampling_type: str = "pad",
    direction: str = "backward",
    **kwargs,
) -> (npt.NDArray, npt.NDArray):
    timeseries_len = len(timeseries)
    input_mask = np.ones(seq_len)

    if timeseries_len >= seq_len:
        return timeseries, input_mask

    if sampling_type == "interpolate":
        timeseries = interpolate_timeseries(timeseries, seq_len)
    elif sampling_type == "pad" and direction == "forward":
        timeseries = np.pad(timeseries, (0, seq_len - timeseries_len), **kwargs)
        input_mask[: seq_len - timeseries_len] = 0
    elif sampling_type == "pad" and direction == "backward":
        timeseries = np.pad(timeseries, ((seq_len - timeseries_len, 0), (0,0)*(len(timeseries.shape)-1)), **kwargs)
        input_mask[: seq_len - timeseries_len] = 0
    else:
        error_msg = "Direction must be one of 'forward' or 'backward'"
        raise ValueError(error_msg)

    assert len(timeseries) == seq_len, "Padding failed"
    return timeseries, input_mask


def downsample_timeseries(
    timeseries: npt.NDArray, seq_len: int, sampling_type: str = "interpolate"
):
    input_mask = np.ones(seq_len)
    if sampling_type == "last":
        timeseries = timeseries[:seq_len]
    elif sampling_type == "first":
        timeseries = timeseries[seq_len:]
    elif sampling_type == "random":
        idx = np.random.randint(0, timeseries.shape[0] - seq_len)
        timeseries = timeseries[idx : idx + seq_len]
    elif sampling_type == "interpolate":
        timeseries = interpolate_timeseries(timeseries, seq_len)
    elif sampling_type == "subsample":
        factor = len(timeseries) // seq_len
        timeseries = timeseries[::factor]
        timeseries, input_mask = upsample_timeseries(
            timeseries, seq_len, sampling_type="pad", direction="forward"
        )
    else:
        error_msg = "Mode must be one of 'last', 'random',\
                'first', 'interpolate' or 'subsample'"
        raise ValueError(error_msg)
    return timeseries, input_mask


def interpolate_timeseries(
    timeseries: npt.NDArray, interp_length: int = 512, channel_first=False,
) -> npt.NDArray:
    if channel_first:
        x = np.linspace(0, 1, timeseries.shape[-1])
        f = interp1d(x, timeseries)
    else:
        x = np.linspace(0, 1, timeseries.shape[0]) 
        f = interp1d(x, timeseries, axis=0)
    x_new = np.linspace(0, 1, interp_length)
    timeseries = f(x_new)
    return timeseries






class MIMIC_mortality(torch.utils.data.Dataset):

    def __init__(self,
                 data_split: str = "train",
                 dir: str = "../data/mimic3_benchmarks/processed/in-hospital-mortality",
                 seed: int = 0,
                 ):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        self.seq_len = 128

        self.data_split = data_split
        self.dir = dir
        self.seed = seed
        self.normalize = True

        self.load_size = None
        if self.small_part:
            if data_split == 'test':
                split = 0.3
            elif data_split == 'train':
                split = 0.6
            else:
                split = 0.1
            self.load_size = int(1000 * split) 

        self._read_data()
        self.filename = dir


    def _read_data(self):

        # Build readers, discretizers, normalizers
        folder = 'test' if self.data_split == 'test' else 'train'
        reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.dir, folder),
                                            listfile=os.path.join(self.dir,
                                                                    f'{self.data_split}_listfile.csv'),
                                            period_length=48.0)

        reader.random_shuffle(self.seed)

        self.discretizer = Discretizer(timestep=1.0,
                                        store_masks=True,
                                        impute_strategy='previous',
                                        start_time='zero',
                                        same_length=False,
                                        timedelta=False)

        self.discretizer_header = self.discretizer.transform(reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(self.discretizer_header) if x.find("->") == -1]

        self.normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = '../data/mimic3_benchmarks/timestamp_ihm_ts-1.00_impute-previous_start-zero_masks-True_n-17903.normalizer'
            
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        self.normalizer.load_params(normalizer_state)

        if not self.normalize:
            self.normalizer = None

        # else:
        self.raw = utils.load_data(reader, self.discretizer, self.normalizer, return_names=True, load_size=self.load_size*2)

        self.data_indices = [i for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.columns = [x for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.n_channels = 60
        self.num_classes = 1

        self.categories = None

        # one hot to ordinal
        self.ordinal_data = []

        used = set()
        mylist =[x.split('->')[0] for x in self.columns]
        unique = [x for x in mylist if x not in used and (used.add(x) or True)]
        vars = [x.split('->')[0] for x in self.columns]

        for i in range(len(self.raw['data'][0])):
            row = []
            for var in unique:
                idx = [i for i, x in enumerate(vars) if x == var]
                if len(idx) == 1:
                    row.append(self.raw['data'][0][i][:, idx[0]])
                else:
                    row.append(np.where(self.raw['data'][0][i][:, idx] == 1)[1])
            row = np.array(row).T
            self.ordinal_data.append(row)

            # get normalizer
            all = np.concatenate(self.ordinal_data, axis=0)
            need_normalize = [x.split('->')[0] for i, x in enumerate(self.columns) if len(x.split('->'))>1 ]
            idxs = set([i for i, x in enumerate(unique) if x in need_normalize])

            # normalize
            for i, d in enumerate(self.ordinal_data):
                for j in idxs:
                    if np.std(all[:, j]) != 0:
                        self.ordinal_data[i][:, j] = (d[:, j] - np.mean(all[:, j])) / np.std(all[:, j])

            self.raw['data'] = list(self.raw['data'])
            self.raw['data'][0] = self.ordinal_data

        self.n_channels = 18
        



    def __getitem__(self, idx):

        data = self.raw['data'][0][idx]
        label = np.array(self.raw['data'][1][idx]).reshape(-1)

        timeseries = data

        timeseries_len = timeseries.shape[0]
        if timeseries_len <= self.seq_len:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len,
                direction='backward',
                sampling_type='pad',
                mode='constant',
            )

        elif timeseries_len > self.seq_len:
            timeseries, input_mask = downsample_timeseries(
                timeseries, self.seq_len, sampling_type='last'
            )

        return timeseries.T, input_mask, label


    def __len__(self):
        return len(self.raw['data'][0])




class MIMIC_phenotyping(torch.utils.data.Dataset):

    def __init__(self,
                 data_split: str = "train",
                 dir: str = "../data/mimic3_benchmarks/processed/phenotyping",
                 seed: int = 0,
                 ):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        self.seq_len = 128

        self.data_split = data_split
        self.dir = dir
        self.seed = seed
        self.normalize = True

        self.load_size = None
        if self.small_part:
            if data_split == 'test':
                split = 0.3
            elif data_split == 'train':
                split = 0.6
            else:
                split = 0.1
            self.load_size = int(1000 * split)

        self._read_data()
        self.filename = dir

        self.is_strats = False


    def _read_data(self):

        # Build readers, discretizers, normalizers
        folder = 'test' if self.data_split == 'test' else 'train'
        reader = PhenotypingReader(dataset_dir=os.path.join(self.dir, folder),
                                            listfile=os.path.join(self.dir,
                                                                    f'{self.data_split}_listfile.csv'),)
        reader.random_shuffle(self.seed)

        self.discretizer = Discretizer(timestep=1.0,
                                        store_masks=True,
                                        impute_strategy='previous',
                                        start_time='zero',
                                        same_length=False,
                                        timedelta=False
                                        )

        self.discretizer_header = self.discretizer.transform(reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(self.discretizer_header) if x.find("->") == -1]

        self.normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize

        normalizer_state = '../data/mimic3_benchmarks/timestamp_pheno_ts-1.00_impute-previous_start-zero_masks-True_n-35621.normalizer'
        
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        self.normalizer.load_params(normalizer_state)

        self.raw = utils.load_data(reader, self.discretizer, self.normalizer, return_names=True, load_size=self.load_size*2)

        self.data_indices = [i for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.columns = [x for (i, x) in enumerate(self.discretizer_header) if x.find("mask") == -1]
        self.n_channels = 60
        self.num_classes = 25

        self.categories = None

        # one hot to ordinal
        self.ordinal_data = []

        used = set()
        mylist =[x.split('->')[0] for x in self.columns]
        unique = [x for x in mylist if x not in used and (used.add(x) or True)]
        vars = [x.split('->')[0] for x in self.columns]

        for i in range(len(self.raw['data'][0])):
            row = []
            for var in unique:
                idx = [i for i, x in enumerate(vars) if x == var]
                if len(idx) == 1:
                    row.append(self.raw['data'][0][i][:, idx[0]])
                else:
                    row.append(np.where(self.raw['data'][0][i][:, idx] == 1)[1])
            row = np.array(row).T
            self.ordinal_data.append(row)

        # get normalizer
        all = np.concatenate(self.ordinal_data, axis=0)
        need_normalize = [x.split('->')[0] for i, x in enumerate(self.columns) if len(x.split('->'))>1 ]
        idxs = set([i for i, x in enumerate(unique) if x in need_normalize])

        # normalize
        for i, d in enumerate(self.ordinal_data):
            for j in idxs:
                if np.std(all[:, j]) != 0:
                    self.ordinal_data[i][:, j] = (d[:, j] - np.mean(all[:, j])) / np.std(all[:, j])

        self.raw['data'] = list(self.raw['data'])
        self.raw['data'][0] = self.ordinal_data

        self.n_channels = 18




    def __getitem__(self, idx):

        data = self.raw['data'][0][idx]
        label = np.array(self.raw['data'][1][idx])


        timeseries = data


        timeseries_len = timeseries.shape[0]
        if timeseries_len <= self.seq_len:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len,
                direction='backward',
                sampling_type='pad',
                mode='constant',
            )

        elif timeseries_len > self.seq_len:
            timeseries, input_mask = downsample_timeseries(
                timeseries, self.seq_len, sampling_type='last'
            )

        return timeseries.T, input_mask, label


    def __len__(self):
        return len(self.raw['data'][0])






