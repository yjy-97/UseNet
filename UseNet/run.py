import argparse
import csv
import os
import torch
from exp.exp_classification import Exp_Classification, RESULT_CSV_COLUMNS
import random
import numpy as np
from datetime import datetime

CV_DATASETS = {"APAVA", "ADFTD", "TDBRAIN", "ADSZ", "MCICN"}


def build_setting(args):
    fold_tag = ""
    if args.data in CV_DATASETS:
        fold_tag = f"_fold{args.fold_index}"

    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        args.seed,
        fold_tag,
    )


def build_summary_setting(args):
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}_5fold_avg".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        args.seed,
    )


def get_fold_indices(args):
    if args.data in CV_DATASETS:
        if args.fold_index is None:
            return list(range(args.num_folds))
        if not 0 <= args.fold_index < args.num_folds:
            raise ValueError(
                f"fold_index must be in [0, {args.num_folds - 1}], got {args.fold_index}"
            )
        return [args.fold_index]
    return [0]


def summarize_fold_results(fold_results):
    if not fold_results:
        return None

    summary = {}
    for split in ["val", "test"]:
        metric_names = fold_results[0][split].keys()
        mean_metrics = {
            metric: np.mean([result[split][metric] for result in fold_results])
            for metric in metric_names
        }
        mean_loss = np.mean([result[f"{split}_loss"] for result in fold_results])
        summary[split] = {"loss": mean_loss, **mean_metrics}
        print(
            f"5-fold average {split.upper()} results --- "
            + ", ".join(
                [f"{k}: {v:.5f}" for k, v in summary[split].items()]
            )
        )
    return summary


def append_result_csv(file_path, rows):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_cv_summary(args, summary_metrics):
    if summary_metrics is None:
        return

    folder_path = (
            "./results/"
            + args.task_name
            + "/"
            + args.model_id
            + "/"
            + args.model
            + "/"
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = "result_classification.txt"
    file_path = os.path.join(folder_path, file_name)
    setting = build_summary_setting(args)
    timestamp = datetime.now().isoformat(timespec="seconds")

    with open(file_path, "a") as f:
        f.write(setting + "  \n")
        for split in ["val", "test"]:
            metrics = summary_metrics[split]
            f.write(
                f"5-fold average {split.upper()} results --- "
                f"Loss: {metrics['loss']:.5f}, "
                f"Accuracy: {metrics['Accuracy']:.5f}, "
                f"Precision: {metrics['Precision']:.5f}, "
                f"Recall: {metrics['Recall']:.5f}, "
                f"F1: {metrics['F1']:.5f}, "
                f"AUROC: {metrics['AUROC']:.5f}, "
                f"AUPRC: {metrics['AUPRC']:.5f}\n"
            )
        f.write("\n")
        f.write("\n")

    common_row = {
        "timestamp": timestamp,
        "setting": setting,
        "task_name": args.task_name,
        "model_id": args.model_id,
        "model": args.model,
        "data": args.data,
        "root_path": args.root_path,
        "fold_index": "avg",
        "num_folds": args.num_folds,
        "is_cv_summary": True,
        "seed": getattr(args, "seed", None),
        "features": args.features,
        "des": args.des,
        "batch_size": args.batch_size,
        "train_epochs": args.train_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "enc_in": getattr(args, "enc_in", None),
        "num_class": getattr(args, "num_class", None),
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "n_heads": args.n_heads,
        "factor": args.factor,
        "dropout": args.dropout,
        "swa": getattr(args, "swa", None),
    }

    rows = []
    for split in ["val", "test"]:
        metrics = summary_metrics[split]
        row = common_row.copy()
        row.update(
            {
                "split": split,
                "loss": metrics["loss"],
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

    global_folder_path = os.path.join("./results/", args.task_name)
    if not os.path.exists(global_folder_path):
        os.makedirs(global_folder_path)
    append_result_csv(
        os.path.join(global_folder_path, "result_classification_all.csv"), rows
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iTransformer Classification")

    parser.add_argument(
        "--task_name",
        type=str,
        default="classification",
        help="task name, fixed to classification",
    )
    parser.add_argument(
        "--is_training", type=int, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UseNet",
        help="model name: iTransformer or UseNet",
    )

    parser.add_argument(
        "--data", type=str, required=True, help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=r"/home/sharedata/shenkaiye/TDBrain",
        help="root path of the data file",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task features",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding",
    )

    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=0, help="prediction sequence length"
    )

    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=6, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=256, help="dimension of fcn")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )

    parser.add_argument("--sampling_rate", type=float, default=128.0,
                        help="EEG sampling rate in Hz; must match the input data")
    parser.add_argument("--virtual_channels", type=int, default=19,
                        help="number of fixed virtual anatomical channels")
    parser.add_argument("--channel_config", type=str, default="",
                        help="JSON file containing source channel names and xyz coordinates")
    parser.add_argument("--channel_names", type=str, default="",
                        help="comma-separated source channel names in exact data order")
    parser.add_argument("--virtual_channel_names", type=str, default="",
                        help="optional comma-separated virtual channel names")
    parser.add_argument("--frequency_bands", type=str,
                        default="1-4,4-8,8-13,13-30,30-45",
                        help="fixed frequency bands in Hz")
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.08,
                        help="entropy regularization used by Sinkhorn OT")
    parser.add_argument("--sinkhorn_iterations", type=int, default=20,
                        help="number of Sinkhorn normalization iterations")
    parser.add_argument("--geometry_weight", type=float, default=1.0)
    parser.add_argument("--signal_weight", type=float, default=0.25)
    parser.add_argument("--quality_weight", type=float, default=0.10)
    parser.add_argument("--confidence_floor", type=float, default=0.05)
    parser.add_argument("--confidence_geometry_temperature", type=float, default=0.35)

    parser.add_argument("--counterfactual_distillation", action="store_true", default=True,
                        help="train with a missing-channel student view")
    parser.add_argument("--no_counterfactual_distillation",
                        action="store_false", dest="counterfactual_distillation",
                        help="disable missing-channel distillation")
    parser.add_argument("--channel_drop_min", type=float, default=0.10)
    parser.add_argument("--channel_drop_max", type=float, default=0.50)
    parser.add_argument("--minimum_keep_channels", type=int, default=4)
    parser.add_argument("--regional_drop_probability", type=float, default=0.50)
    parser.add_argument("--distillation_temperature", type=float, default=2.0)
    parser.add_argument("--student_ce_weight", type=float, default=1.0)
    parser.add_argument("--lambda_kd", type=float, default=0.50)
    parser.add_argument("--lambda_mapping", type=float, default=0.10)
    parser.add_argument("--lambda_band", type=float, default=0.10)

    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument(
        "--num_folds", type=int, default=5, help="number of folds for cross validation"
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        default=None,
        help="run a specific fold only; default runs all folds for CV datasets",
    )
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="Exp", help="exp description")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging",
        default=False,
    )

    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=4, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="1,4", help="device ids of multiple gpus"
    )

    args = parser.parse_args()
    args.task_name = "classification"  # force classification
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_Classification
    fold_indices = get_fold_indices(args)

    if args.is_training:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            fold_results = []
            for fold_idx in fold_indices:
                args.seed = seed
                args.fold_index = fold_idx
                setting = build_setting(args)

                exp = Exp(args)
                print(
                    ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>".format(
                        setting
                    )
                )
                exp.train(setting)

                print(
                    ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                metrics = exp.test(setting)
                if metrics is not None:
                    fold_results.append(metrics)
            torch.cuda.empty_cache()

            if len(fold_indices) > 1:
                summary_metrics = summarize_fold_results(fold_results)
                write_cv_summary(args, summary_metrics)
    else:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            fold_results = []
            for fold_idx in fold_indices:
                args.seed = seed
                args.fold_index = fold_idx
                setting = build_setting(args)

                exp = Exp(args)
                print(
                    ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                metrics = exp.test(setting, test=1)
                if metrics is not None:
                    fold_results.append(metrics)
            torch.cuda.empty_cache()

            if len(fold_indices) > 1:
                summary_metrics = summarize_fold_results(fold_results)
                write_cv_summary(args, summary_metrics)
