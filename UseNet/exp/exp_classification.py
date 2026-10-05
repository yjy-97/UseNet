import csv
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


RESULT_CSV_COLUMNS = [
    "timestamp",
    "setting",
    "task_name",
    "model_id",
    "model",
    "data",
    "root_path",
    "split",
    "fold_index",
    "num_folds",
    "is_cv_summary",
    "seed",
    "features",
    "des",
    "batch_size",
    "train_epochs",
    "learning_rate",
    "patience",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "num_class",
    "d_model",
    "d_ff",
    "e_layers",
    "d_layers",
    "n_heads",
    "factor",
    "dropout",
    "swa",
    "loss",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "AUPRC",
]


def append_result_csv(file_path, rows):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len
        self.args.pred_len = 0
        source_channels = int(test_data.X.shape[2])
        self.args.source_enc_in = source_channels
        if self.args.model == "UseNet":
            self.args.enc_in = int(self.args.virtual_channels)
        else:
            self.args.enc_in = source_channels
        self.args.num_class = len(np.unique(test_data.y))
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                loss = criterion(outputs, label.long())
                total_loss.append(float(loss.detach().cpu()))

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.args.model == "UseNet":
                    model_output = self.model(
                        batch_x, padding_mask, None, None, return_aux=True
                    )
                    if isinstance(model_output, dict):
                        outputs = model_output["logits"]
                        student_outputs = model_output["student_logits"]
                        aux_losses = model_output["aux_losses"]
                        loss_full = criterion(outputs, label.long())
                        loss_student = criterion(student_outputs, label.long())
                        loss = (
                            loss_full
                            + self.args.student_ce_weight * loss_student
                            + self.args.lambda_kd * aux_losses["kd"].mean()
                            + self.args.lambda_mapping * aux_losses["mapping"].mean()
                            + self.args.lambda_band * aux_losses["band"].mean()
                        )
                    else:
                        outputs = model_output
                        loss = criterion(outputs, label.long())
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()

        timestamp = datetime.now().isoformat(timespec="seconds")
        fold_index = (
            getattr(self.args, "fold_index", None)
            if self.args.data in {"APAVA", "ADFTD", "TDBRAIN", "ADSZ", "MCICN"}
            else None
        )

        common_row = {
            "timestamp": timestamp,
            "setting": setting,
            "task_name": self.args.task_name,
            "model_id": self.args.model_id,
            "model": self.args.model,
            "data": self.args.data,
            "root_path": self.args.root_path,
            "fold_index": fold_index,
            "num_folds": getattr(self.args, "num_folds", None),
            "is_cv_summary": False,
            "seed": getattr(self.args, "seed", None),
            "features": self.args.features,
            "des": self.args.des,
            "batch_size": self.args.batch_size,
            "train_epochs": self.args.train_epochs,
            "learning_rate": self.args.learning_rate,
            "patience": self.args.patience,
            "seq_len": self.args.seq_len,
            "label_len": self.args.label_len,
            "pred_len": self.args.pred_len,
            "enc_in": self.args.enc_in,
            "num_class": self.args.num_class,
            "d_model": self.args.d_model,
            "d_ff": self.args.d_ff,
            "e_layers": self.args.e_layers,
            "d_layers": self.args.d_layers,
            "n_heads": self.args.n_heads,
            "factor": self.args.factor,
            "dropout": self.args.dropout,
            "swa": self.swa,
        }

        rows = []
        for split, loss, metrics in [
            ("val", vali_loss, val_metrics_dict),
            ("test", test_loss, test_metrics_dict),
        ]:
            row = common_row.copy()
            row.update(
                {
                    "split": split,
                    "loss": float(loss),
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1": metrics["F1"],
                    "AUROC": metrics["AUROC"],
                    "AUPRC": metrics["AUPRC"],
                }
            )
            rows.append(row)

        append_result_csv(os.path.join(folder_path, "result_classification.csv"), rows)

        global_folder_path = os.path.join("./results/", self.args.task_name)
        if not os.path.exists(global_folder_path):
            os.makedirs(global_folder_path)
        append_result_csv(
            os.path.join(global_folder_path, "result_classification_all.csv"), rows
        )

        return {
            "val": val_metrics_dict,
            "test": test_metrics_dict,
            "val_loss": float(vali_loss),
            "test_loss": float(test_loss),
        }
