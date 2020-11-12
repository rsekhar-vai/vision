import os
import re
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

def fit_model(data_train, data_val, model, loss_func=None,
              optimizer=None, scheduler=None, args=None):
    args, tracker = setup_for_fit_model(args)
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
    batches_train = torch.utils.data.DataLoader(data_train, args.batch_size,
                                          shuffle=True)
    batches_val = torch.utils.data.DataLoader(data_val, args.batch_size,
                                          shuffle=True)
    model = model.to(args.device)
    try:
        for epoch_index in range(args.num_epochs):
            tracker['epoch_index'] = epoch_index
            print("--------------------- @epoch ", epoch_index, "---------------------")
            running_loss = 0.0
            running_acc = 0.0
            model.train()
            for batch_index, batch_dict in enumerate(batches_train):
                data = batch_dict[0].to(args.device)
                target = batch_dict[1].to(args.device)
                optimizer.zero_grad()
                prediction = model(data)
                loss = loss_func(prediction, target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                loss.backward()
                optimizer.step()
                acc_t = compute_accuracy(prediction, target)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            tracker['train_loss'].append(running_loss)
            tracker['train_acc'].append(running_acc)
            print('  training loss/accuracy {:.5f} / {:.2f}'.format(running_loss, running_acc))

            running_loss = 0.
            running_acc = 0.
            model.eval()
            for batch_index, batch_dict in enumerate(batches_val):
                data = batch_dict[0].to(args.device)
                target = batch_dict[1].to(args.device)
                prediction = model(data)
                loss = loss_func(prediction, target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(prediction, target)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            tracker['val_loss'].append(running_loss)
            tracker['val_acc'].append(running_acc)
            print('validation loss/accuracy {:.5f} / {:.2f}'.format(running_loss, running_acc))

            tracker = update_tracker(args=args, model=model,
                                     tracker=tracker)

            scheduler.step(tracker['val_loss'][-1])

            if tracker['stop_early']:
                break

    except KeyboardInterrupt:
        print("Exiting loop")

    return tracker

def setup_for_fit_model(args):
    if args is None:
        args = Namespace(
            model_state_file="model.pth",
            save_dir="model_storage/Clf",
            seed=1337,
            learning_rate=0.001,
            dropout_p=0.1,
            batch_size=16,
            num_epochs=20,
            early_stopping_criteria=5,
            cuda=True,
            catch_keyboard_interrupt=True,
            reload_from_files=False,
            expand_filepaths_to_save_dir=True,
        )
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)
    print("Expanded filepaths: ")
    print("\t{}".format(args.model_state_file))
    if not torch.cuda.is_available():
        args.cuda = False
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))
    set_seed_everywhere(args.seed, args.cuda)
    tracker = init_tracker(args)

    return args, tracker

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def init_tracker(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}


def update_tracker(args, model, tracker):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param tracker: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if tracker['epoch_index'] == 0:
        torch.save(model.state_dict(), tracker['model_filename'])
        tracker['stop_early'] = False

    # Save model if performance improved
    elif tracker['epoch_index'] >= 1:
        loss_tm1, loss_t = tracker['val_loss'][-2:]

        # If loss worsened
        if loss_t >= tracker['early_stopping_best_val']:
            # Update step
            tracker['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < tracker['early_stopping_best_val']:
                torch.save(model.state_dict(), tracker['model_filename'])

            # Reset early stopping step
            tracker['early_stopping_step'] = 0

        # Stop early ?
        tracker['stop_early'] = \
            tracker['early_stopping_step'] >= args.early_stopping_criteria

    return tracker

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100
