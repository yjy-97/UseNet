import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_provider.uea import (
    normalize_batch_ts,
)
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
try:
    from natsort import natsorted
except ImportError:
    import re

    def natsorted(values):
        def natural_key(value):
            return [
                int(token) if token.isdigit() else token.lower()
                for token in re.split(r"(\d+)", str(value))
            ]
        return sorted(values, key=natural_key)

warnings.filterwarnings("ignore")



class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)
        all_ids = np.asarray(data_list[:, 1]).astype(int)
        all_labels = np.asarray(data_list[:, 0])
        self.fold_index = int(getattr(args, "fold_index", getattr(args, "fold", 0)))
        n_splits = 5

        if not 0 <= self.fold_index < n_splits:
            raise ValueError(
                f"fold_index must be in [0, {n_splits - 1}], got {self.fold_index}"
            )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = [test_idx for _, test_idx in skf.split(all_ids, all_labels)]

        test_subject_idx = fold_indices[self.fold_index]
        val_subject_idx = fold_indices[(self.fold_index + 1) % n_splits]
        train_subject_idx = np.setdiff1d(
            np.arange(len(all_ids)),
            np.concatenate([test_subject_idx, val_subject_idx]),
        )

        self.train_ids = all_ids[train_subject_idx].tolist()
        self.val_ids = all_ids[val_subject_idx].tolist()
        self.test_ids = all_ids[test_subject_idx].tolist()

        self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_apava(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train ids:", ids)
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("val ids:", ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test ids:", ids)
        else:
            ids = set(np.asarray(subject_label[:, 1]).astype(int).tolist())
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            subject_id = int(trial_label[1])
            for trial_feature in subject_feature:
                if subject_id in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class TDBRAINLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)
        all_ids = np.asarray(data_list[:, 1]).astype(int)
        all_labels = np.asarray(data_list[:, 0])
        self.fold_index = int(getattr(args, "fold_index", getattr(args, "fold", 0)))
        n_splits = 5

        if not 0 <= self.fold_index < n_splits:
            raise ValueError(
                f"fold_index must be in [0, {n_splits - 1}], got {self.fold_index}"
            )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = [test_idx for _, test_idx in skf.split(all_ids, all_labels)]

        test_subject_idx = fold_indices[self.fold_index]
        val_subject_idx = fold_indices[(self.fold_index + 1) % n_splits]
        train_subject_idx = np.setdiff1d(
            np.arange(len(all_ids)),
            np.concatenate([test_subject_idx, val_subject_idx]),
        )

        self.train_ids = all_ids[train_subject_idx].tolist()
        self.val_ids = all_ids[val_subject_idx].tolist()
        self.test_ids = all_ids[test_subject_idx].tolist()

        self.X, self.y = self.load_tdbrain(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_tdbrain(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train ids:", ids)
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("val ids:", ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test ids:", ids)
        else:
            ids = set(np.asarray(subject_label[:, 1]).astype(int).tolist())
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            subject_id = int(trial_label[1])
            for trial_feature in subject_feature:
                if subject_id in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)
        all_ids = np.asarray(data_list[:, 1]).astype(int)
        all_labels = np.asarray(data_list[:, 0])
        self.fold_index = int(getattr(args, "fold_index", getattr(args, "fold", 0)))
        n_splits = 5

        if not 0 <= self.fold_index < n_splits:
            raise ValueError(
                f"fold_index must be in [0, {n_splits - 1}], got {self.fold_index}"
            )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = [test_idx for _, test_idx in skf.split(all_ids, all_labels)]

        test_subject_idx = fold_indices[self.fold_index]
        val_subject_idx = fold_indices[(self.fold_index + 1) % n_splits]
        train_subject_idx = np.setdiff1d(
            np.arange(len(all_ids)),
            np.concatenate([test_subject_idx, val_subject_idx]),
        )

        self.train_ids = all_ids[train_subject_idx].tolist()
        self.val_ids = all_ids[val_subject_idx].tolist()
        self.test_ids = all_ids[test_subject_idx].tolist()
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_adfd(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train ids:", ids)
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("val ids:", ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test ids:", ids)
        else:
            ids = set(np.asarray(subject_label[:, 1]).astype(int).tolist())
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            subject_id = int(trial_label[1])
            for trial_feature in subject_feature:
                if subject_id in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.55, 0.7

        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptb(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        my_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                if j + 1 in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBXLLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptbxl(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        no_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        mi_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        sttc_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])
        cd_list = list(data_list[np.where(data_list[:, 0] == 3)][:, 1])
        hyp_list = list(data_list[np.where(data_list[:, 0] == 4)][:, 1])

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_ptbxl(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                if j + 1 in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADSZLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)
        all_ids = np.asarray(data_list[:, 1]).astype(int)
        all_labels = np.asarray(data_list[:, 0])
        self.fold_index = int(getattr(args, "fold_index", getattr(args, "fold", 0)))
        n_splits = 5

        if not 0 <= self.fold_index < n_splits:
            raise ValueError(
                f"fold_index must be in [0, {n_splits - 1}], got {self.fold_index}"
            )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = [test_idx for _, test_idx in skf.split(all_ids, all_labels)]

        test_subject_idx = fold_indices[self.fold_index]
        val_subject_idx = fold_indices[(self.fold_index + 1) % n_splits]
        train_subject_idx = np.setdiff1d(
            np.arange(len(all_ids)),
            np.concatenate([test_subject_idx, val_subject_idx]),
        )

        self.train_ids = all_ids[train_subject_idx].tolist()
        self.val_ids = all_ids[val_subject_idx].tolist()
        self.test_ids = all_ids[test_subject_idx].tolist()
        self.X, self.y = self.load_data(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_data(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train ids:", ids)
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("val ids:", ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test ids:", ids)
        else:
            ids = set(np.asarray(subject_label[:, 1]).astype(int).tolist())
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            subject_id = int(trial_label[1])
            for trial_feature in subject_feature:
                if subject_id in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class MCICNLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)
        all_ids = np.asarray(data_list[:, 1]).astype(int)
        all_labels = np.asarray(data_list[:, 0])
        self.fold_index = int(getattr(args, "fold_index", getattr(args, "fold", 0)))
        n_splits = 5

        if not 0 <= self.fold_index < n_splits:
            raise ValueError(
                f"fold_index must be in [0, {n_splits - 1}], got {self.fold_index}"
            )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = [test_idx for _, test_idx in skf.split(all_ids, all_labels)]

        test_subject_idx = fold_indices[self.fold_index]
        val_subject_idx = fold_indices[(self.fold_index + 1) % n_splits]
        train_subject_idx = np.setdiff1d(
            np.arange(len(all_ids)),
            np.concatenate([test_subject_idx, val_subject_idx]),
        )

        self.train_ids = all_ids[train_subject_idx].tolist()
        self.val_ids = all_ids[val_subject_idx].tolist()
        self.test_ids = all_ids[test_subject_idx].tolist()
        self.X, self.y = self.load_data(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_data(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train ids:", ids)
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("val ids:", ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test ids:", ids)
        else:
            ids = set(np.asarray(subject_label[:, 1]).astype(int).tolist())
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            subject_id = int(trial_label[1])
            for trial_feature in subject_feature:
                if subject_id in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)

