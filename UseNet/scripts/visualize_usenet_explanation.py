from __future__ import annotations

import argparse
import os
import random
import re
import sys
from argparse import Namespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn
from exp.exp_classification import Exp_Classification
from utils.usenet_interpretability import compute_usenet_explanation, explanation_to_dict
from utils.usenet_visualization import save_usenet_explanation_report


CV_DATASETS = {"APAVA", "ADFTD", "TDBRAIN", "ADSZ", "MCICN"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def infer_setting_from_checkpoint(checkpoint: str) -> Dict[str, object]:
    """Infer common run.py arguments from the checkpoint parent folder name."""
    setting = os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))
    pattern = re.compile(
        r"^(?P<task_name>[^_]+)_"
        r"(?P<model_id>.+?)_"
        r"(?P<model>iTransformer|UseNet)_"
        r"(?P<data>[^_]+)_"
        r"ft(?P<features>.*?)_"
        r"sl(?P<seq_len>\d+)_"
        r"ll(?P<label_len>\d+)_"
        r"pl(?P<pred_len>\d+)_"
        r"dm(?P<d_model>\d+)_"
        r"nh(?P<n_heads>\d+)_"
        r"el(?P<e_layers>\d+)_"
        r"dl(?P<d_layers>\d+)_"
        r"df(?P<d_ff>\d+)_"
        r"fc(?P<factor>\d+)_"
        r"eb(?P<embed>.*?)_"
        r"dt(?P<distil>True|False)_"
        r"(?P<des>.*?)_"
        r"seed(?P<seed>\d+)"
        r"(?:_fold(?P<fold_index>\d+))?$"
    )
    match = pattern.match(setting)
    if not match:
        return {}

    raw = match.groupdict()
    parsed: Dict[str, object] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if key in {
            "seq_len",
            "label_len",
            "pred_len",
            "d_model",
            "n_heads",
            "e_layers",
            "d_layers",
            "d_ff",
            "factor",
            "seed",
            "fold_index",
        }:
            parsed[key] = int(value)
        elif key == "distil":
            parsed[key] = value == "True"
        else:
            parsed[key] = value
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VEUS-iTransformer interpretation visualizations."
    )

    parser.add_argument("--checkpoint", type=str, default=r'/home/yejiayu/Work/Project/eeg/checkpoints/classification/test/UseNet/classification_test_UseNet_TDBRAIN_ftM_sl256_ll48_pl0_dm128_nh8_el6_dl1_df256_fc1_ebtimeF_dtTrue_Exp_seed41_fold4/checkpoint.pth', help="Path to checkpoint.pth")
    parser.add_argument("--root_path", type=str, default=r"/home/sharedata/shenkaiye/TDBrain", help="Dataset root path containing Feature/ and Label/")
    parser.add_argument("--sample_npy", type=str, default="", help="Optional single EEG sample .npy file shaped [T,C] or [1,T,C]")
    parser.add_argument("--data", type=str, default="ADFTD", help="Dataset name, e.g. ADFTD, APAVA, TDBRAIN")
    parser.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"], help="Dataset split")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index within the selected split")
    parser.add_argument("--output_dir", type=str, default="./usenet_vis", help="Directory for saved figures")
    parser.add_argument(
        "--mode", "--vis_mode",
        dest="mode",
        type=str,
        default="source",
        choices=["source", "virtual"],
        help="Visualization space: source for original EEG channels, virtual for unified virtual regions."
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-k bars/channels shown in auxiliary plots")
    parser.add_argument("--target_class", type=int, default=None, help="Class to explain; default explains predicted class")
    parser.add_argument("--no_occlusion", action="store_true", help="Disable virtual-channel occlusion attribution for speed")
    parser.add_argument("--show_names", action="store_true", help="Show channel names on topomap points")

    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--model", type=str, default="UseNet")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=0)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=6)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true", default=False)

    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--virtual_channels", type=int, default=19)
    parser.add_argument("--channel_config", type=str, default="")
    parser.add_argument("--channel_names", type=str, default="")
    parser.add_argument("--virtual_channel_names", type=str, default="")
    parser.add_argument("--frequency_bands", type=str, default="1-4,4-8,8-13,13-30,30-45")
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.08)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--geometry_weight", type=float, default=1.0)
    parser.add_argument("--signal_weight", type=float, default=0.25)
    parser.add_argument("--quality_weight", type=float, default=0.10)
    parser.add_argument("--confidence_floor", type=float, default=0.05)
    parser.add_argument("--confidence_geometry_temperature", type=float, default=0.35)

    parser.add_argument("--counterfactual_distillation", action="store_true", default=True)
    parser.add_argument("--channel_drop_min", type=float, default=0.10)
    parser.add_argument("--channel_drop_max", type=float, default=0.50)
    parser.add_argument("--minimum_keep_channels", type=int, default=4)
    parser.add_argument("--regional_drop_probability", type=float, default=0.50)
    parser.add_argument("--distillation_temperature", type=float, default=2.0)
    parser.add_argument("--student_ce_weight", type=float, default=1.0)
    parser.add_argument("--lambda_kd", type=float, default=0.50)
    parser.add_argument("--lambda_mapping", type=float, default=0.10)
    parser.add_argument("--lambda_band", type=float, default=0.10)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--swa", action="store_true", default=False)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    return parser


def apply_inferred_defaults(args: Namespace, inferred: Dict[str, object]) -> Namespace:
    """Fill unset or default fields with values inferred from the checkpoint path."""
    defaults = build_parser().parse_args([
        "--checkpoint", args.checkpoint
    ])
    for key, value in inferred.items():
        if not hasattr(args, key):
            continue
        current_value = getattr(args, key)
        default_value = getattr(defaults, key, None)
        if current_value == default_value or current_value in {"", None}:
            setattr(args, key, value)
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_state_dict_robust(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint object type: {type(state)}")

    model_state = model.state_dict()
    has_module_in_ckpt = any(k.startswith("module.") for k in state.keys())
    has_module_in_model = any(k.startswith("module.") for k in model_state.keys())
    if has_module_in_ckpt and not has_module_in_model:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    elif has_module_in_model and not has_module_in_ckpt:
        state = {"module." + k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def load_sample_from_dataset(args: Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_set, _ = data_provider(args, args.split)
    if len(data_set) == 0:
        raise RuntimeError(f"No samples found in split {args.split}.")
    index = int(args.sample_index)
    if index < 0 or index >= len(data_set):
        raise IndexError(f"sample_index={index} is out of range for split {args.split}, length={len(data_set)}")
    batch_x, label, padding_mask = collate_fn([data_set[index]], max_len=args.seq_len)
    return batch_x, label, padding_mask


def load_sample_from_npy(args: Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arr = np.load(args.sample_npy)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"sample_npy must be [T,C] or [1,T,C], got shape {arr.shape}")
    x = torch.from_numpy(arr).float()
    if x.shape[1] > args.seq_len:
        x = x[:, : args.seq_len, :]
    elif x.shape[1] < args.seq_len:
        pad = torch.zeros(x.shape[0], args.seq_len - x.shape[1], x.shape[2], dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    padding_mask = torch.zeros(x.shape[0], args.seq_len, dtype=torch.bool)
    valid_len = min(arr.shape[1], args.seq_len)
    padding_mask[:, :valid_len] = True
    label = torch.zeros(x.shape[0], dtype=torch.long)
    return x, label, padding_mask


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Warning: ignored unknown arguments:", unknown)

    inferred = infer_setting_from_checkpoint(args.checkpoint)
    if inferred:
        args = apply_inferred_defaults(args, inferred)
        print("Inferred checkpoint setting:", inferred)
    else:
        print("Warning: could not infer model settings from checkpoint path; using CLI/default values.")

    if not args.data:
        raise ValueError("Please provide --data, e.g. --data ADFTD")
    if not args.root_path and not args.sample_npy:
        raise ValueError(
            "Please provide --root_path for the dataset, or --sample_npy for a single preprocessed sample. "
            "The checkpoint path alone does not contain EEG data."
        )

    args.is_training = 0
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    args.batch_size = 1
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(i) for i in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    seed_everything(int(args.seed))

    exp = Exp_Classification(args)
    load_state_dict_robust(exp.model, args.checkpoint, exp.device)
    exp.model.eval()

    if args.sample_npy:
        batch_x, label, padding_mask = load_sample_from_npy(args)
    else:
        batch_x, label, padding_mask = load_sample_from_dataset(args)

    batch_x = batch_x.float().to(exp.device)
    padding_mask = padding_mask.float().to(exp.device)

    explanation = compute_usenet_explanation(
        model=exp.model,
        x=batch_x,
        time_mask=padding_mask,
        target_class=args.target_class,
        normal_stats=None,
        use_occlusion=not args.no_occlusion,
    )

    paths = save_usenet_explanation_report(
        explanation=explanation,
        sample_index=0,
        output_dir=args.output_dir,
        mode=args.mode,
        top_k=args.top_k,
        show_names=args.show_names,
    )

    summary = explanation_to_dict(explanation, sample_index=0, top_k=min(args.top_k, 10))
    print("Label:", int(label.reshape(-1)[0].item()))
    print("Explanation summary:", summary)
    print("Saved visualization files:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
