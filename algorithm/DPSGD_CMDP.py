import copy

import torch

from data.util.sampling import get_data_loaders_possion
from privacy_analysis.RDP.compute_cmdp import epsilon_R_subsampling, rdp_to_epsilon_dp
from train_and_validation.train import train
from train_and_validation.train_with_dp import  train_with_dp, train_with_dp_cmdp
from train_and_validation.validation import validation
from algorithm.CMDP_Mechanism import hyper_para_setup


def DPSGD_CMDP(train_data, test_data, model,optimizer, hyper_para, batch_size, epsilon_budget, device, log_file):

    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')


    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size,microbatch_size=1,iterations=1)

    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    iter = 1
    epsilon_con = 0.
    best_test_acc=0.
    RDP_con = 0.
    epsilon_list=[]
    test_loss_list=[]
    epsilon_total = epsilon_budget
    epsilon_used = hyper_para.epsilon
    N_epoch = 10
    lr_used = hyper_para.lr

    while epsilon_con < epsilon_total:
        train_dl = minibatch_loader(train_data)  # possion sampling
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp_cmdp(model, train_dl, optimizer, epsilon_used, lr_used, device)

        test_loss, test_accuracy = validation(model, test_dl, device)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter

        RDP_con_iter = optimizer.epsilon_R_subsampling(batch_size / len(train_data))

        RDP_con += RDP_con_iter

        epsilon_con = optimizer.rdp_to_epsilon_dp(RDP_con, epsilon_con)

        epsilon_list.append(torch.tensor(epsilon_con))
        test_loss_list.append(test_loss)

        log_message(f'iters:{iter},epsilon:{epsilon_con:.4f} | Test set: Average loss: {test_loss:.4f}, Accuracy:({test_accuracy:.2f}%)')

        lr_new, epsilon_new = hyper_para.hyper_parameter_update(test_accuracy, test_loss, lr_used, epsilon_used, iter, N_epoch)
        if (lr_new != None) and (epsilon_new != None):
            lr_used = lr_new
            epsilon_used = epsilon_new

        iter += 1

    log_message("------finished ------")
    return test_accuracy,iter,best_test_acc,best_iter,model,[epsilon_list,test_loss_list]
