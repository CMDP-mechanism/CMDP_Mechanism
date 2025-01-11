from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader
from model.CNN import  CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_max_steps, get_min_sigma
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.dp_utils import scatter_normalization
from utils.dp_optimizer import DPSGD_Optimizer, DPAdam_Optimizer
import torch

from train_and_validation.train_with_dp import train_with_dp, train_with_dp_cmdp
from train_and_validation.validation import validation
import copy
import numpy as np

from data.util.sampling import  get_data_loaders_possion

from data.util.dividing_validation_data import dividing_validation_set, dividing_validation_set_for_IMDB
import os
from algorithm.CMDP_Mechanism import hyper_para_setup


def DPSUR_CMDP(dataset_name,train_dataset, test_data, model, batch_size, hyper_para, momentum, epsilon_budget, C_t, sigma_t,use_scattering,input_norm,bn_noise_multiplier,num_groups,bs_valid,C_v,beta,sigma_v,MIA, device, log_file):

    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    rdp_norm = 0.

    #if MIA==True, Do not using scatter
    if MIA:
        train_data=train_dataset
        if dataset_name != 'IMDB':
            optimizer = DPSGD_Optimizer(
                l2_norm_clip=C_t,
                noise_multiplier=sigma_t,
                minibatch_size=batch_size,
                microbatch_size=1,
                params=model.parameters(),
                lr=hyper_para.lr,
                momentum=momentum
            )
        else:
            optimizer = DPAdam_Optimizer(
                l2_norm_clip=C_t,
                noise_multiplier=sigma_t,
                minibatch_size=batch_size,
                microbatch_size=1,
                params=model.parameters(),
                lr=hyper_para.lr)
    else:
        if dataset_name != 'IMDB':

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)


            if use_scattering:
                scattering, K, _ = get_scatter_transform(dataset_name)
                scattering.to(device)
            else:
                scattering = None
                K = 3 if len(train_dataset.data.shape) == 4 else 1

            if input_norm == "BN":
                save_dir = f"bn_stats/{dataset_name}"
                os.makedirs(save_dir, exist_ok=True)
                bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                           scattering,
                                                           K,
                                                           device,
                                                           len(train_dataset),
                                                           len(train_dataset),
                                                           noise_multiplier=bn_noise_multiplier,
                                                           save_dir=save_dir)
                model = CNNS[dataset_name](K, input_norm="BN", bn_stats=bn_stats, size=None)


            else:
                model = CNNS[dataset_name](K, input_norm=input_norm, num_groups=num_groups, size=None)

            model.to(device)
            train_data = get_scattered_dataset(train_loader, scattering, device, len(train_dataset))
            test_dl = get_scattered_loader(test_dl, scattering, device)

            optimizer = DPSGD_Optimizer(
                l2_norm_clip=C_t,
                noise_multiplier=sigma_t,
                minibatch_size=batch_size,
                microbatch_size=1,
                params=model.parameters(),
                lr=hyper_para.lr,
                momentum=momentum
            )
        else:
            optimizer = DPAdam_Optimizer(
                l2_norm_clip=C_t,
                noise_multiplier=sigma_t,
                minibatch_size=batch_size,
                microbatch_size=1,
                params=model.parameters(),
                lr=hyper_para.lr)

    minibatch_loader_for_train, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size, microbatch_size=1, iterations=1)
    minibatch_loader_for_valid, microbatch_loader = get_data_loaders_possion(minibatch_size=bs_valid, microbatch_size=1, iterations=1)

    last_valid_loss = 100000.0
    last_accept_test_acc=0.
    last_model = model
    t = 1
    iter=1
    best_iter=1
    best_test_acc=0.

    epsilon_list=[]
    test_loss_list=[]
    epsilon_used = hyper_para.epsilon
    N_epoch = 10
    lr_used = hyper_para.lr
    RDP_con = 0.
    epsilon_con = 0.
    epsilon_total = epsilon_budget

    while epsilon_con < epsilon_total:
    # batch_size / len(train_dataset)
    # bs_valid / len(train_dataset)

        if dataset_name=='IMDB':
            train_dl = minibatch_loader_for_train(train_dataset)
            valid_dl = minibatch_loader_for_valid(train_dataset)
            for id, (data, target) in enumerate(train_dl):
                optimizer.minibatch_size = len(data)

        else:
            train_dl = minibatch_loader_for_train(train_data)
            valid_dl = minibatch_loader_for_valid(train_data)
            for id, (data, target) in enumerate(train_dl):
                optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp_cmdp(model, train_dl, optimizer, epsilon_used, lr_used, device)

        RDP_con_iter_train = optimizer.epsilon_R_subsampling(batch_size / len(train_dataset))

        valid_loss, valid_accuracy = validation(model, valid_dl, device)

        RDP_con_iter_valid = optimizer.epsilon_R_subsampling(bs_valid / len(train_dataset))

        test_loss, test_accuracy = validation(model, test_dl, device)

        RDP_con = RDP_con + RDP_con_iter_train + RDP_con_iter_valid

        epsilon_con = optimizer.rdp_to_epsilon_dp(RDP_con, epsilon_con)

        deltaE=valid_loss - last_valid_loss
        deltaE=torch.tensor(deltaE).cpu()
        print("Delta E:",deltaE)

        deltaE= np.clip(deltaE,-C_v,C_v)
        deltaE_after_dp =2*C_v*sigma_v*np.random.normal(0,1)+deltaE

        print("Delta E after dp:",deltaE_after_dp)

        if deltaE_after_dp < beta*C_v:
            last_valid_loss = valid_loss
            last_model = copy.deepcopy(model)
            t = t + 1
            print("accept updates，the number of updates t：", format(t))
            last_accept_test_acc=test_accuracy

            if last_accept_test_acc > best_test_acc:
                best_test_acc = last_accept_test_acc
                best_iter = t

            epsilon_list.append(torch.tensor(epsilon_con))
            test_loss_list.append(test_loss)

        else:
            print("reject updates")
            model.load_state_dict(last_model.state_dict(), strict=True)

        log_message(f'iters:{iter},epsilon:{epsilon_con:.4f} | Test set: Average loss: {test_loss:.4f}, Accuracy:({test_accuracy:.2f}%)')

        lr_new, epsilon_new = hyper_para.hyper_parameter_update(test_accuracy, test_loss, lr_used, epsilon_used, iter, N_epoch)
        if (lr_new != None) and (epsilon_new != None):
            lr_used = lr_new
            epsilon_used = epsilon_new

        iter+=1

    log_message("------finished ------")
    return last_accept_test_acc,t,best_test_acc,best_iter,last_model,[epsilon_list,test_loss_list]

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}
