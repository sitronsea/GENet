# Main function. Arguments.
import importlib
import argparse

import torch

import json
import wandb
import os
import sys

from . import train
from . import eval
from . import utils


def parse_args():
    desc = (
        "Train a gauge equivariant neural network "
        "for outputting the Berry curvature."
    )
    parser = argparse.ArgumentParser(description=desc)

    # Seed
    parser.add_argument("--seed", type=int, default=None)

    # Logging
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)

    # Mode
    parser.add_argument(
        "--quantity",
        choices=["local", "global"],
        default="train",
        help="Whether to learn the local or the global quantities"
    )

    # Data
    parser.add_argument(
        "--samples",
        type=int,
        default=1024,
        help="The number of samples generated per epoch"
    )
    parser.add_argument(
        "--n_bands",
        type=int,
        default=4,
        help="The number of filled bands"
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="*",
        default=[5, 5],
        help="The size of the lattice; length must be even"
    )
    parser.add_argument(
        "--label_mode",
        choices=["Berry", "Wilson"],
        default="Berry",
        help="Wilson label or Berry label",
    )
    parser.add_argument(
        "--keep_only_trivial_samples",
        action="store_true",
        help="Keep only trivial samples"
    )
    parser.add_argument(
        "--diag_ratio",
        type=float,
        default=0.,
        help="The ratio of diagonal samples; only valid for 2D lattices"
    )
    parser.add_argument(
        "--label_distribution",
        type=float,
        default=None,
        nargs="*",
        help="The distribution of phase angles of the fluxes"
    )

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["GEBLNet", "GEConvNet", "TrFCNet", 'DeepEig'],
        default="GEBLNet"
    )
    parser.add_argument(
        "--layer_channels",
        type=int,
        default=[32, 16, 8],
        nargs="*",
        help="The layer output channels of the model",
    )    
    parser.add_argument(
        "--deepchannels",
        type=int,
        default=[32, 16, 8],
        nargs="*",
        help="The layer output channels of the model",
    )    
    parser.add_argument(
        "--eigchannels",
        type=int,
        default=[32, 16, 8],
        nargs="*",
        help="The layer output channels of the model",
    )
    parser.add_argument(
        "--use_unit_elements",
        type=utils.str_to_bool,
        default=True,
        help="Use unit matrices for GEBL and GEConv"
    )
    parser.add_argument(
        "--conjugation_enlargement",
        type=utils.str_to_bool,
        default=True,
        help="Conjugation enlargements for GEBL and GEConv. Accepts 'true' or 'false'."
    )

    parser.add_argument(
        "--kernel_size",
        type=int,
        default=4.,
        help="Kernel size of Conv Layers"
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=4,
        help="Dilation step of Conv Layers"
    )
    parser.add_argument(
        "--normalize_trace",
        type=utils.str_to_bool,
        default=True,
        help="Normalization of the Trace layer based on a fixed hyperparameter"
    )
    parser.add_argument(
        "--trnorm",
        action="store_true",
        help="Implement the TrNorm layers"
    )
    parser.add_argument("--trnorm_threshold", default=1e-3)
    parser.add_argument("--trnorm_before_ReLU", action="store_true")
    parser.add_argument("--trnorm_on_abs", action="store_true")

    # Parameter initialization
    parser.add_argument(
        "--init_weight_factor",
        type=float,
        default=1.,
        help="The init_weight_factor"
    )
    parser.add_argument(
        "--init_variant",
        type=int,
        choices=[-1, 0, 1, 2, 4],
        default=0
    )

    # Training
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="The learning rate"
    )
    parser.add_argument("--batch", type=int, default=32, help="The batch size")
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=5000,
        help="Total number of epochs"
    )
    parser.add_argument(
        "--std_clamp",
        type=float,
        default=0.5,
        help="The std clamping value for the std loss"
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["step", "exp", "cyclic"],
        default="step"
    )
    parser.add_argument(
        "--lr_schedule_milestones",
        type=float,
        default=[1],
        nargs="*",
        help="Percentages at which the learning rate is rescheduled"
    )

    # Evaluating
    parser.add_argument(
        "--hamiltonian_samples",
        action="store_true"
    )
    parser.add_argument(
        "--rescale_eval",
        action="store_true",
        help="Whether to rescale outputs for evaluation"
    )

    # Saving
    # model
    parser.add_argument(
        "--save_folder_nets",
        type=str,
        default="models_GEN/",
        help="The folder containing network dicts"
    )
    parser.add_argument(
        "--save_model_name",
        type=str,
        default="GEN_.pt",
        help="Network file name",
    )
    parser.add_argument(
        "--load_model_name",
        type=str,
        default=None,
        help="Network file name",
    )
    parser.add_argument(
        "--model_save_frequency",
        type=int,
        default=10,
        help="save model frequency"
    )
    # eval results
    parser.add_argument(
        "--save_folder_eval",
        type=str, default="eval/",
        help="The output files"
    )
    parser.add_argument(
        "--save_eval_name",
        type=str,
        default="GEN_",
        help="The name of saved histogram plots",
    )

    parser.add_argument("--output_dir", type=str, default="./")

    return parser.parse_args()


def main():
    """
    MAIN PROCEDURE
    """
    args = parse_args()

    if args.seed is None:
        args.seed = torch.randint(1000, (1,))

    torch.manual_seed(args.seed)

    with importlib.resources.path("gauge_net", "config.json") as config_path:
        print(config_path)  # config_path is a pathlib.Path object

    with open(config_path, 'r') as f:
        config = json.load(f)

    wandb_config = getattr(config, "wandb", None)
    # WandB outputs
    wandb_run = wandb.init(
        project=getattr(wandb_config, "project", None),
        entity=getattr(wandb_config, "entity", None),
        name=getattr(wandb_config, "name", args.name),
        config=args.__dict__,
        dir=getattr(wandb_config, "dir", os.getenv("WANDB_OUTPUT", "./")),
    )
    args.name = wandb_run.name
    
    command = sys.argv[0]

    if "train" in command:
        save_name, save_extension = os.path.splitext(args.save_model_name)
        args.save_model_name = f"{save_name}_{args.name}{save_extension}"
    elif "eval" in command:
        args.save_eval_name += "_" + args.name

    try:
        if "train" in command:
            train.train(args)

        elif "eval" in command:
            eval.eval(args)

        else:
            print("Unknown command. Use 'gauge_net_train' or 'gauge_net_eval'.")
            sys.exit(1)

    except Exception:
        wandb.finish(exit_code=1)
        raise

    wandb.finish(exit_code=0)


if __name__ == "__main__":
    main()
