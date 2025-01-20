import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_layers", required=True, type=int)
# parser.add_argument("--layer_width", required=True, type=int)
# args = parser.parse_args()

# n_layers = args.n_layers
# layer_width = args.layer_width

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
import pickle

from data_processing.datasets import PigletDataset, HelicoidDataset, GivenDataset
from neuralnet.model import SpectraMLP
import config
from torch.utils.data import ConcatDataset
from utils import *
from data_processing import *
from data_processing.generate_dataset import *
from tqdm.auto import tqdm
import time
import timeit
import gc

def run_test_set_helicoid(hyperparams, test_set_patients=["025-02"], cuda="cuda:0", batch_size=1000):
    config.molecule_count = 10
    print(os.getcwd())
    #config.dataset_path = "/home/kevin/piglets_helicoid/idp-beerinverse/dataset/helicoid"
    train_on_synth=hyperparams["train_on_synth"]
    n_layers=hyperparams["n_layers"]
    layer_width=hyperparams["layer_width"]
    act = hyperparams["act"]
    #min_max_values=hyperparams["min_max_values"]
    
    if train_on_synth:
        train_mode_path = "synthetic_scattering"
    else:
        train_mode_path = "real_scattering"
    
    name = train_mode_path + "_" + str(n_layers) + "_" + str(layer_width)
    
    error_test_set = []
    cumulative_error = 0
    n=0
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    for patient in test_set_patients:
        path = config.dataset_path + 'helicoid/'
        
        test_set = 2
        dl = DataLoader(test_set, batch_size=len(test_set),shuffle=False)
        
        model = SpectraMLP(config.molecule_count, n_layers=n_layers, layer_width=layer_width, act_fc=act)
        checkpoints = torch.load('./results/' + name +'/best_model.pth')
        model.load_state_dict(checkpoints)
        model.to(device)
        model.eval()
        
        for i, (inputs, targets) in enumerate(tqdm(dl)):
            assert(i==0)
            #if i==0:
            inputs = inputs.to(device).float()
            inputs = torch.squeeze(inputs)
            outputs = model(inputs)
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            plot_helicoid_pred(targets, outputs, name, test_set.getimgsizes(), patient=patient)
            
            error = np.mean(np.abs(outputs - targets))
            cumulative_error += error
            error_test_set.append(error)
            
            plt.savefig('./results/' + name + '/' + patient+"_"+str(i)+".png")
            
        n += 1
            
    print(n)
    #assert(n == 4)        
    cumulative_error = cumulative_error / n
    hyperparams["cumulative_error"] = cumulative_error
    hyperparams["error_test_set"] = error_test_set
    hyperparams["test_set_patients"] = test_set_patients
    with open('./results/' + name + '/params.pkl', 'wb') as f:
        pickle.dump(hyperparams, f)
    return cumulative_error

def run_patient_helicoid(hyperparams, patient="025-02", cuda="cuda:0", batch_size=1000, report_time=False):
    config.molecule_count = 10
    print(os.getcwd())
    #config.dataset_path = "/home/kevin/piglets_helicoid/idp-beerinverse/dataset/helicoid"
    train_on_synth=hyperparams["train_on_synth"]
    n_layers=hyperparams["n_layers"]
    layer_width=hyperparams["layer_width"]
    act = hyperparams["act"]
    #min_max_values=hyperparams["min_max_values"]
    
    if train_on_synth:
        train_mode_path = "synthetic_scattering"
    else:
        train_mode_path = "real_scattering"
    
    name = train_mode_path + "_" + str(n_layers) + "_" + str(layer_width)
    
    error_test_set = []
    cumulative_error = 0
    n=0
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    path = config.dataset_path + 'helicoid/'
    
    test_set = HelicoidDataset(path, patients=[patient], min_max_values=None)
    dl = DataLoader(test_set, batch_size=len(test_set),shuffle=False)
    
    model = SpectraMLP(config.molecule_count, n_layers=n_layers, layer_width=layer_width, act_fc=act)
    checkpoints = torch.load('./results/' + name +'/best_model.pth')
    model.load_state_dict(checkpoints)
    model.to(device)
    model.eval()
    
    assert len(dl) == 1

    inputs, targets = next(iter(dl))
    
    start_dl = time.time()
    inputs = inputs.to(device).float()
    inputs = torch.squeeze(inputs)
    
    start_calc = time.time()
    outputs = model(inputs)
    end_calc = time.time()

    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    end_dl = time.time()
    if report_time:
        return outputs, targets, end_calc-start_calc, end_dl - start_dl
    else:
        return outputs, targets
        
    # n += 1
            
    # print(n)
    # #assert(n == 4)        
    # cumulative_error = cumulative_error / n
    # hyperparams["cumulative_error"] = cumulative_error
    # hyperparams["error_test_set"] = error_test_set
    # hyperparams["test_set_patients"] = test_set_patients
    # with open('./results/' + name + '/params.pkl', 'wb') as f:
    #     pickle.dump(hyperparams, f)
    # return cumulative_error

# def run_data_with_timings(hyperparams, attenuation, params, number_runs, use_gpu): #runs on cpu the given data and returns timing
    
#     def net_comp(model, inputs):
#         def f():
#             return model(inputs)
#         return f

#     print(os.getcwd())
#     n_layers=hyperparams["n_layers"]
#     layer_width=hyperparams["layer_width"]
#     act = hyperparams["act"]
    
#     config.scattering_model = hyperparams["scattering_model"]
#     config.train_on_synthetic = hyperparams["train_on_synthetic"]
#     scatter_name = "scatter" if config.scattering_model else "no_scatter"
#     train_on_synthetic_name = "train_synth" if config.train_on_synthetic else "train_real"
#     name = scatter_name + "_" + train_on_synthetic_name + "_" + str(n_layers) + "_" + str(layer_width)
    
#     cumulative_error = 0
#     n=0
    
#     if use_gpu:
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")
    
#     if config.scattering_model:
#         path = config.dataset_path+"piglet_scattering"
#         model_name = "scattering"
#     else:
#         path = config.dataset_path+"piglet_no_scattering"
#         model_name = "non scattering"

#     if config.train_on_synthetic:
#         train_name = "synthetic"
#     else:
#         train_name = "real"

#     print(path)
#     dataset = GivenDataset(attenuation, params)
#     dl = DataLoader(dataset, batch_size=len(dataset),shuffle=False) #everything in one batch so maximum parallelization
    
#     assert(len(dl)==1)
    
#     model = SpectraMLP(3, n_layers=n_layers, layer_width=layer_width, act_fc=act)
#     checkpoints = torch.load('./results_piglets/' + name +'/best_model.pth')
#     model.load_state_dict(checkpoints)
#     model.to(device)
#     model.eval()
    
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
#     coef = ["HbO2", "Hbb", "diffCCO"]
#     for i, (inputs, targets) in enumerate(tqdm(dl)):
#         #if i==0:
#         inputs = inputs.to(device).float()
#         inputs = torch.squeeze(inputs)
#         # outputs = model(inputs)
#         # outputs = outputs.detach().cpu().numpy()
#         # targets = targets.detach().cpu().numpy()
        
#         runtime = timeit.timeit(net_comp(model,inputs),number=number_runs)/number_runs
#         # plt.figure()
#         # for j in range(outputs.shape[1]):
#         #     plt.plot(targets[:,j], color=colors[j], label=f'Opti ' + coef[j], linewidth=3, alpha=0.5)
#         #     plt.plot(outputs[:,j], color=colors[j], label=f'NN ' + coef[j], linewidth=0.5)                   
#         # legend = plt.legend()
#         # error = np.sum(np.abs(outputs-targets)) / outputs.shape[0]
#         # cumulative_error += error
#         # n += 1
#         # plt.xlim((0,2000))
#         # plt.savefig('./results/' + name + '/' + "timing.png")
    
#     del model
#     del inputs
#     torch.cuda.empty_cache()
#     gc.collect()
            
#     return runtime

def run_piglet(hyperparams, piglet):
    print(os.getcwd())
    n_layers=hyperparams["n_layers"]
    layer_width=hyperparams["layer_width"]
    act = hyperparams["act"]
    
    config.scattering_model = hyperparams["scattering_model"]
    config.train_on_synthetic = hyperparams["train_on_synthetic"]
    scatter_name = "scatter" if config.scattering_model else "no_scatter"
    train_on_synthetic_name = "train_synth" if config.train_on_synthetic else "train_real"
    name = scatter_name + "_" + train_on_synthetic_name + "_" + str(n_layers) + "_" + str(layer_width)
    
    error_test_set = []
    cumulative_error = 0
    n=0
    device = torch.device("cpu")

    test_range = range(piglet, piglet+1)

    # if config.train_on_synthetic:
    #     #test_range = range(475, 512)
    #     test_range = range(507, 513)
    # else:
    #     test_range = range(507, 513)

    for piglet in test_range:
        try:
            if config.scattering_model:
                path = config.dataset_path+"piglet_scattering"
                model_name = "scattering"
            else:
                path = config.dataset_path+"piglet_no_scattering"
                model_name = "non scattering"

            if config.train_on_synthetic:
                train_name = "synthetic"
            else:
                train_name = "real"

            print(path)
            test_set = PigletDataset(path, range=(piglet, piglet))
            dl = DataLoader(test_set, batch_size=1000,shuffle=False)
            
            #model = SpectraMLP(3, n_layers=1, layer_width=128)
            #checkpoints = torch.load('./results/1_128/best_model.pth')
            model = SpectraMLP(3, n_layers=n_layers, layer_width=layer_width, act_fc=act)
            checkpoints = torch.load('./results_piglets/' + name +'/best_model.pth')
            model.load_state_dict(checkpoints)
            model.to(device)
            model.eval()
            
            inputs, targets = next(iter(dl))
            assert len(dl) == 1
            #print(inputs,targets)
            inputs = inputs.to(device).float()
            inputs = torch.squeeze(inputs)
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            #plt.figure()
            # print(outputs.shape)
            # print(targets.shape)
            return outputs, targets
        except:
            print("No piglet "+str(piglet))

def run_test_set(hyperparams):
    print(os.getcwd())
    n_layers=hyperparams["n_layers"]
    layer_width=hyperparams["layer_width"]
    act = hyperparams["act"]
    
    config.scattering_model = hyperparams["scattering_model"]
    config.train_on_synthetic = hyperparams["train_on_synthetic"]
    scatter_name = "scatter" if config.scattering_model else "no_scatter"
    train_on_synthetic_name = "train_synth" if config.train_on_synthetic else "train_real"
    name = scatter_name + "_" + train_on_synthetic_name + "_" + str(n_layers) + "_" + str(layer_width)
    
    error_test_set = []
    cumulative_error = 0
    n=0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config.train_on_synthetic:
        #test_range = range(475, 512)
        test_range = range(507, 513)
    else:
        test_range = range(507, 513)

    for piglet in test_range:
        try:
            if config.scattering_model:
                path = config.dataset_path+"piglet_scattering"
                model_name = "scattering"
            else:
                path = config.dataset_path+"piglet_no_scattering"
                model_name = "non scattering"

            if config.train_on_synthetic:
                train_name = "synthetic"
            else:
                train_name = "real"

            print(path)
            test_set = PigletDataset(path, range=(piglet, piglet))
            dl = DataLoader(test_set, batch_size=1000,shuffle=False)
            
            #model = SpectraMLP(3, n_layers=1, layer_width=128)
            #checkpoints = torch.load('./results/1_128/best_model.pth')
            model = SpectraMLP(3, n_layers=n_layers, layer_width=layer_width, act_fc=act)
            checkpoints = torch.load('./results_piglets/' + name +'/best_model.pth')
            model.load_state_dict(checkpoints)
            model.to(device)
            model.eval()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            coef = ["HbO2", "Hbb", "diffCCO"]
            for i, (inputs, targets) in enumerate(tqdm(dl)):
                #if i==0:
                inputs = inputs.to(device).float()
                inputs = torch.squeeze(inputs)
                outputs = model(inputs)
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                plt.figure()
                #print(outputs.shape)
                #print(targets.shape)
                for j in range(outputs.shape[1]):
                    plt.plot(targets[:,j], color=colors[j], label=f'Opti ' + coef[j], linewidth=3, alpha=0.5)
                    plt.plot(outputs[:,j], color=colors[j], label=f'NN ' + coef[j], linewidth=0.5)                   
                legend = plt.legend()
                error = np.sum(np.abs(outputs-targets)) / outputs.shape[0]
                cumulative_error += error
                error_test_set.append(error)
                n += 1
                plt.title("Piglet " + str(piglet) + " with " + model_name + " model trained on " + train_name + " with (" + str(n_layers) + "," + str(layer_width) + ")" +"\nL1 error = {:.4e}".format(error))
                plt.savefig('./results_piglets/' + name + '/' + str(piglet)+"_"+str(i)+".png")
        except:
            print("No piglet "+str(piglet))
            
    print(n)
    print(len(test_range))
    assert(n == 4)        
    cumulative_error = cumulative_error / n
    hyperparams["cumulative_error"] = cumulative_error
    hyperparams["error_test_set"] = error_test_set
    with open('./results_piglets/' + name + '/params.pkl', 'wb') as f:
        pickle.dump(hyperparams, f)
    return cumulative_error

def run_data_with_timings(hyperparams, attenuation, params, number_runs, use_gpu): #runs on cpu the given data and returns timing
    
    def net_comp(model, inputs):
        def f():
            return model(inputs)
        return f

    #print(os.getcwd())
    n_layers=hyperparams["n_layers"]
    layer_width=hyperparams["layer_width"]
    act = hyperparams["act"]
    
    config.scattering_model = hyperparams["scattering_model"]
    config.train_on_synthetic = hyperparams["train_on_synthetic"]
    scatter_name = "scatter" if config.scattering_model else "no_scatter"
    train_on_synthetic_name = "train_synth" if config.train_on_synthetic else "train_real"
    name = scatter_name + "_" + train_on_synthetic_name + "_" + str(n_layers) + "_" + str(layer_width)
    
    cumulative_error = 0
    n=0
    
    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    if config.scattering_model:
        path = config.dataset_path+"piglet_scattering"
        model_name = "scattering"
    else:
        path = config.dataset_path+"piglet_no_scattering"
        model_name = "non scattering"

    if config.train_on_synthetic:
        train_name = "synthetic"
    else:
        train_name = "real"

    #print(path)
    dataset = GivenDataset(attenuation, params)
    dl = DataLoader(dataset, batch_size=len(dataset),shuffle=False) #everything in one batch so maximum parallelization
    
    assert(len(dl)==1)
    
    model = SpectraMLP(3, n_layers=n_layers, layer_width=layer_width, act_fc=act)
    checkpoints = torch.load('./results_piglets/' + name +'/best_model.pth')
    model.load_state_dict(checkpoints)
    model.to(device)
    model.eval()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    coef = ["HbO2", "Hbb", "diffCCO"]
    for i, (inputs, targets) in enumerate(dl):
        #if i==0:
        assert(i==0)
        inputs = inputs.to(device).float()
        inputs = torch.squeeze(inputs)
        # outputs = model(inputs)
        # outputs = outputs.detach().cpu().numpy()
        # targets = targets.detach().cpu().numpy()
        
        runtime = timeit.timeit(net_comp(model,inputs),number=number_runs)/number_runs
        # plt.figure()
        # for j in range(outputs.shape[1]):
        #     plt.plot(targets[:,j], color=colors[j], label=f'Opti ' + coef[j], linewidth=3, alpha=0.5)
        #     plt.plot(outputs[:,j], color=colors[j], label=f'NN ' + coef[j], linewidth=0.5)                   
        # legend = plt.legend()
        # error = np.sum(np.abs(outputs-targets)) / outputs.shape[0]
        # cumulative_error += error
        # n += 1
        # plt.xlim((0,2000))
        # plt.savefig('./results/' + name + '/' + "timing.png")
    
    del model
    del inputs
    torch.cuda.empty_cache()
    gc.collect()
            
    return runtime
