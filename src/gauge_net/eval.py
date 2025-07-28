# Evaluation
import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import wandb

from . import data
from . import model
from . import utils


def analyze_outputs(
    labels,
    local_outputs,
    eval_folder,
    save_name
):
    # # Calculate MAE
    # print("LOCAL MAE ERROR: ")
    # MAE_error = np.mean(np.abs(labels.reshape(-1) - local_outputs.reshape(-1)))
    # print(MAE_error)

    # print("MEAN ABSOLUTE LOCAL LABEL: ")
    # mean_abs_local_label = np.mean(np.abs(labels.reshape(-1)))
    # print(mean_abs_local_label)

    # Calculate global output
    global_labels = labels
    global_outputs = np.sum(local_outputs, 1)
    global_outputs_rounded = np.round(global_outputs)

    correct = (global_outputs_rounded == np.round(global_labels))
    wrong = (global_outputs_rounded != np.round(global_labels))

    data_wrong = np.stack(
        [
            np.round(global_labels[wrong]),
            global_outputs[wrong],
            np.round(global_labels[wrong]) - global_outputs_rounded[wrong]
        ], axis=1
    )

    # Log wrong outputs
    table_wrong = wandb.Table(
        data=data_wrong,
        columns=["chern_labels", "chern_outputs", "difference"]
    )
    wandb.log({"chern_comparison_wrong": table_wrong})

    Correct_samples = np.sum(correct)
    Wrong_samples = np.sum(wrong)
    print("Correct: ", Correct_samples, "Wrong: ", Wrong_samples)

    wandb.log({
        # "local_mae": MAE_error,
        # "mean_abs_local_label": mean_abs_local_label,
        # "relative_mae": MAE_error / mean_abs_local_label,
        "chern_acc": Correct_samples / (Correct_samples + Wrong_samples)
    })


def eval(args):
    # loaders
    if args.hamiltonian_samples:
        test_dataset = data.HamiltonianDataset(args)
    else:
        test_dataset = data.ProjectDataset(args)
    test_loader = DataLoader(test_dataset, args.batch, num_workers=4)

    # the model
    model_type = getattr(model, args.model_type)
    net = model_type(args)

    # CUDA for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # load the network
    model_folder = args.output_dir + args.save_folder_nets
    net.load_state_dict(
        torch.load(os.path.join(model_folder, args.save_model_name))
    )

    eval_folder = os.path.join(args.output_dir, args.save_folder_eval)

    # test
    net.eval()
    ibatch = 0
    data_save_dir = "./saved_data"
    os.makedirs(data_save_dir, exist_ok=True)
    with torch.no_grad():
        for local_batch, local_labels in test_loader:
            print("Batch: ", ibatch)
            ibatch += 1

            # Compute outputs
            local_batch, local_labels = (
                local_batch.to(device),
                local_labels.to(device)
            )

            labels = torch.sum(local_labels, 1)
            local_outputs = net(local_batch)
            outputs = local_outputs
            if args.rescale_eval:
                outputs = utils.rescale(local_outputs, local_labels)

            # save
            if ibatch == 1:
                total_labels = labels.clone()
                total_outputs = outputs.clone()
            else:
                total_labels = torch.cat((total_labels, labels), 0)
                total_outputs = torch.cat((total_outputs, outputs), 0)

        local_outputs = total_outputs.cpu().detach().numpy()
        labels = total_labels.cpu().detach().numpy()

        # eval folder
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        analyze_outputs(
            labels,
            local_outputs,
            eval_folder,
            args.save_eval_name
        )
