import os
import time

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import utils
from data_processing.datasets import PigletDataset, SpectraDataset
from neuralnet.model import SpectraMLP
import config

from ray import tune, train


# def train_piglet(n_layers=4, layer_width=1024, batch_size=128):
#     """
#     created to train on datasets in dataset/piglet_diffs
#     results are saved in the results folder, for each n_layers/layer_width setup seperately
#     """
    
#     print("Training on synthetic: " + str(config.train_on_synthetic))
#     print("Scattering model: " + str(config.scattering_model))
    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     lr = config.lr

#     name = str(n_layers) + "_" + str(layer_width)
#     if not os.path.exists('results/{}'.format(name)):
#         os.mkdir("results/{}".format(name))
#     save_path = 'results/{}/best_model.pth'.format(name)

#     model = SpectraMLP(config.molecule_count, n_layers=n_layers, layer_width=layer_width)
#     model.to(device)
#     criterion = MSELoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     loss = 0.0

#     #train_set = PigletDataset(config.dataset_path+"piglet_diffs", range=(475, 503))
#     if config.train_on_synthetic:
#         if config.scattering_model:
#             print("Training on synthetic scattering model")
#             train_set = SpectraDataset(config.dataset_path+"synthetic_scattering/")
#         else:
#             train_set = SpectraDataset(config.dataset_path+"synthetic_no_scattering/")
#     else:
#         if config.scattering_model:
#             train_set = PigletDataset(config.dataset_path+"piglet_scattering", range=(475, 503))
#         else:
#             train_set = PigletDataset(config.dataset_path+"piglet_no_scattering", range=(475, 503))
    
#     if config.scattering_model:
#         val_set = PigletDataset(config.dataset_path+"piglet_scattering", range=(504, 504))
#     else:
#         val_set = PigletDataset(config.dataset_path+"piglet_no_scattering", range=(504, 504))


#     train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     val_dl = DataLoader(val_set, batch_size=64, shuffle=False)

#     best_loss, best_age = 100000, 0
#     ground_truth, predictions = None, None
#     start = time.time()
#     for epoch in range(config.epochs):
#         print('Epoch {}/{}'.format(epoch + 1, config.epochs))
#         print('-' * 10)
#         model.train()
#         # Iterate through training data loader
#         for i, (inputs, targets) in enumerate(tqdm(train_dl)):
#             inputs, targets = inputs.to(device).float(), targets.to(device).float()
#             inputs = torch.squeeze(inputs)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             loss.backward()
#             optimizer.step()

#         if epoch == 0:
#             print(model.summary())

#         val_loss = 0
#         model.eval()
#         ground_truth, predictions = None, None
#         for i, (inputs, targets) in enumerate(tqdm(val_dl)):
#             inputs, targets = inputs.to(device).float(), targets.to(device).float()
#             inputs = torch.squeeze(inputs)
#             outputs = model(inputs)
#             if ground_truth is None:
#                 ground_truth = targets
#                 predictions = outputs
#             else:
#                 ground_truth = torch.cat((ground_truth, targets), 0)
#                 predictions = torch.cat((predictions, outputs), 0)

#             val_loss += criterion(outputs, targets)
#         val_loss = val_loss / len(val_dl)

#         print("validation loss at epoch {}:".format(epoch), val_loss.item())
#         if best_loss > val_loss:
#             torch.save(model.state_dict(), save_path)
#             utils.plot_pred(ground_truth, predictions, name)
#             best_loss = val_loss
#             best_age = 0
#         else:
#             best_age += 1
#             print("patience: {}/{}".format(best_age, config.patience))
#             if best_age == int(config.patience / 2):
#                 print("50% patience reached, decrease learning rate")
#                 lr /= 10
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = lr
#                     print("New learning rate: ", param_group['lr'])
#             if best_age >= config.patience:
#                 print("patience reached, stop training")
#                 break


#     time_delta = time.time() - start
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_delta // 60, time_delta % 60
#     ))

#     print(ground_truth[100])
#     print(predictions[100])


def train_raytune_piglet(hyperparams):
    """
    created to train on datasets in dataset/piglet_diffs
    results are saved in the results folder, for each n_layers/layer_width setup seperately
    """
    os.chdir(config.path)
    #print(os.getcwd())
    n_layers=hyperparams["n_layers"]
    layer_width=hyperparams["layer_width"]
    batch_size=hyperparams["batch_size"]
    lr = hyperparams["lr"]
    act = hyperparams["act"]
    
    config.scattering_model = hyperparams["scattering_model"]
    config.train_on_synthetic = hyperparams["train_on_synthetic"]
    save_model = hyperparams["save_model"]
    
    print(n_layers)
    print(layer_width)
    
    print("Training on synthetic: " + str(config.train_on_synthetic))
    print("Scattering model: " + str(config.scattering_model))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scatter_name = "scatter" if config.scattering_model else "no_scatter"
    train_on_synthetic_name = "train_synth" if config.train_on_synthetic else "train_real"

    if save_model:
        name = scatter_name + "_" + train_on_synthetic_name + "_" + str(n_layers) + "_" + str(layer_width)
        if not os.path.exists('results_piglets'):
            os.mkdir("results_piglets")
        if not os.path.exists('results_piglets/{}'.format(name)):
            os.mkdir("results_piglets/{}".format(name))
        save_path = 'results_piglets/{}/best_model.pth'.format(name)

    model = SpectraMLP(config.molecule_count, n_layers=n_layers, layer_width=layer_width, act_fc=act)
    model.to(device)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    loss = 0.0

    #train_set = PigletDataset(config.dataset_path+"piglet_diffs", range=(475, 503))
    if config.train_on_synthetic:
        if config.scattering_model:
            print("Training on synthetic scattering model")
            train_set = SpectraDataset(config.dataset_path+"synthetic_scattering/")
        else:
            train_set = SpectraDataset(config.dataset_path+"synthetic_no_scattering/")
    else:
        if config.scattering_model:
            train_set = PigletDataset(config.dataset_path+"piglet_scattering", range=(475, 502))
        else:
            train_set = PigletDataset(config.dataset_path+"piglet_no_scattering", range=(475, 502))
        assert(len(train_set) == 19*1000)
    
    if config.scattering_model:
        val_set = PigletDataset(config.dataset_path+"piglet_scattering", range=(503, 504))
    else:
        val_set = PigletDataset(config.dataset_path+"piglet_no_scattering", range=(503, 504))

    print("train set len", len(train_set))
    assert(len(val_set) == 2*1000)

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    best_loss, best_age = np.inf, 0
    ground_truth, predictions = None, None
    start = time.time()
    for epoch in range(config.epochs):
        #print('Epoch {}/{}'.format(epoch + 1, config.epochs))
        #print('-' * 10)
        model.train()
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate((train_dl)):
            #print(inputs.shape)
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            inputs = torch.squeeze(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        if epoch == 0:
            print(model.summary())

        val_loss = 0
        model.eval()
        ground_truth, predictions = None, None
        for i, (inputs, targets) in enumerate((val_dl)):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            inputs = torch.squeeze(inputs)
            outputs = model(inputs)
            if ground_truth is None:
                ground_truth = targets
                predictions = outputs
            else:
                ground_truth = torch.cat((ground_truth, targets), 0)
                predictions = torch.cat((predictions, outputs), 0)

            val_loss += criterion(outputs, targets)
        val_loss = val_loss / len(val_dl)

        #print("validation loss at epoch {}:".format(epoch), val_loss.item())
        if best_loss > val_loss:
            if save_model:
                torch.save(model.state_dict(), save_path)
                utils.plot_pred(ground_truth, predictions, name)
            best_loss = val_loss
            best_age = 0
        else:
            best_age += 1
            #print("patience: {}/{}".format(best_age, config.patience))
            if best_age == int(config.patience / 2):
                print("50% patience reached, decrease learning rate")
                lr /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    #print("New learning rate: ", param_group['lr'])
            if best_age >= config.patience:
                #print("patience reached, stop training")
                break


    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_delta // 60, time_delta % 60
    ))

    #print(ground_truth[100])
    #print(predictions[100])
    
    train.report({"loss":best_loss.item()})
    
    #return val_loss

# nn_params = [(4, 1024), (3, 1024), (2, 1024), (1, 1024),
#              (4, 512), (3, 512), (2, 512), (1, 512),
#              (4, 256), (3, 256), (2, 256), (1, 256),
#              (4, 128), (3, 128), (2, 128), (1, 128)]

#nn_params = [(1, 128)]
# nn_params = [(1,128)]

# for p in nn_params:
#     train_piglet(n_layers=p[0], layer_width=p[1])

# config.scattering_model = True
# config.train_on_synthetic=False

# analysis = tune.run(
#     train_raytune_piglet,
#     config={
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([32, 64, 128]),
#         "n_layers": tune.qrandint(0,5),
#         "layer_width": tune.sample_from(lambda spec: 0 if spec.config.n_layers == 0 else tune.qrandint(1,256))
#     },
#     num_samples=1  # Number of times to sample from the hyperparameter space
# )