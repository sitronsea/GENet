# Training
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR

import os
import wandb

from . import model
from . import data
from . import utils


def get_lr_scheduler(scheduler, optimizer, start_lr, total_epochs, milestones):
    if scheduler == "step":
        milestone_epochs = [round(total_epochs * ms) for ms in milestones]
        print(milestone_epochs)
        return MultiStepLR(
            optimizer,
            milestone_epochs,
            gamma=0.1
        )

    elif scheduler == "exp":
        # factor by which lr has reduced at the end of training:
        final_reduction_factor = 0.01
        gamma = final_reduction_factor ** (1 / total_epochs)
        return ExponentialLR(
            optimizer,
            gamma
        )

    elif scheduler == "cyclic":
        amplitude_factor = 0.01
        base_lr = start_lr * amplitude_factor
        cycles = 2
        step_size_up = int((total_epochs / cycles) // 2)
        return CyclicLR(
            optimizer,
            base_lr,
            start_lr,
            step_size_up,
            cycle_momentum=False
        )


def train(args):
    # Data loader
    # Training data
    train_dataset = data.ProjectDataset(args)
    train_loader = DataLoader(
        train_dataset,
        args.batch,
        num_workers=4
    )

    # Testing data
    test_dataset = data.ProjectDataset(args)
    test_loader = DataLoader(test_dataset, args.batch, num_workers=4)

    # Create the model output folder
    model_folder = os.path.join(args.output_dir, args.save_folder_nets)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # The model
    net_type = getattr(model, args.net_type)
    net = net_type(args)

    # CUDA for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # optimizer and criterion
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = get_lr_scheduler(
        args.lr_schedule,
        optimizer,
        args.lr,
        args.total_epochs,
        args.lr_schedule_milestones
    )

    # loop over the epochs
    for i_epoch in range(args.total_epochs):

        print("-------   Epoch ", i_epoch, " ----------------")

        # train
        net.train()
        current_batch = 0
        for local_batch, local_labels in train_loader:
            # data
            local_batch, local_labels = (
                local_batch.to(device),
                local_labels.to(device)
            )
            # outputs
            local_outputs = net(local_batch)
            if args.quantity == "global":
                # global criterion (compared with the trivial class)
                global_outputs = torch.sum(local_outputs, 1)
                global_labels = torch.sum(local_labels, 1)
                global_loss = criterion(global_outputs, global_labels)

                loss = global_loss

                if args.keep_only_trivial_samples:
                    # std criterior (to force the output to locally differ)
                    std_outputs = torch.clamp(
                        torch.std(local_outputs, 1),
                        max=args.std_clamp
                    )
                    std_goal = torch.ones_like(std_outputs) * args.std_clamp
                    std_loss = criterion(std_outputs, std_goal)

                    loss += std_loss

            else:
                loss = criterion(local_outputs, local_labels)

            loss.backward()

            # optimize
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            current_batch += 1

        scheduler.step()

        # test
        net.eval()
        running_loss_local = 0
        running_loss_local_rescaled = 0
        running_loss_global = 0
        running_loss_std = 0
        running_std_test = 0
        running_loss_global = 0

        with torch.no_grad():
            for local_batch, local_labels in test_loader:

                # test
                local_batch, local_labels = (
                    local_batch.to(device),
                    local_labels.to(device)
                )

                # Get outputs
                local_outputs = net(local_batch)
                global_outputs = torch.sum(
                    local_outputs,
                    1
                )
                std_outputs = torch.std(local_outputs, 1)
                local_outputs_rescaled = utils.rescale(
                    local_outputs,
                    local_labels
                )
                # Calculate losses
                local_loss = criterion(
                    local_outputs,
                    local_labels
                )
                local_loss_rescaled = criterion(
                    local_outputs_rescaled,
                    local_labels
                )
                global_loss = criterion(
                    global_outputs,
                    global_labels
                )

                std_goal = torch.ones_like(std_outputs) * args.std_clamp
                std_loss = criterion(
                    std_goal,
                    torch.clamp(
                        std_outputs,
                        max=args.std_clamp
                    )
                )

                # print statistics
                running_loss_local += local_loss.item()
                running_loss_local_rescaled += local_loss_rescaled.item()
                running_loss_global += global_loss.item()
                running_loss_std += std_loss.item()
                running_std_test += std_outputs.item()

            mean_loss_local = running_loss_local / len(test_loader)
            mean_loss_local_rescaled = running_loss_local_rescaled / len(test_loader)
            mean_loss_global = running_loss_global / len(test_loader)
            mean_std = running_std_test / len(test_loader)
            mean_std_loss = running_loss_std / len(test_loader)
            mean_total_loss = (running_loss_global + running_loss_std) / len(test_loader)

            print(
                "Epoch [%d]\n" % (i_epoch + 1),
                "local test loss: %.6f\n" % (mean_loss_local),
                "local rescaled test loss: %.6f\n" % (mean_loss_local_rescaled),
                "global test loss: %.6f\n" % (mean_loss_global),
                "std test loss: %.6f\n" % (mean_std_loss),
                "total test loss: %.6f\n" % (mean_total_loss)
            )
        wandb.log(
            {
                "test_loss_local": mean_loss_local,
                "test_loss_local_rescaled": mean_loss_local_rescaled,
                "test_loss_global": mean_loss_global,
                "test_std_loss": mean_std_loss,
                "test_total_loss": mean_total_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "test_mean_std": mean_std,
            }
        )

        # save the model
        if (i_epoch + 1) % args.model_save_frequency == 0:
            filename = os.path.join(model_folder, args.save_model_name)
            torch.save(net.state_dict(), filename)
            wandb.save(filename)
