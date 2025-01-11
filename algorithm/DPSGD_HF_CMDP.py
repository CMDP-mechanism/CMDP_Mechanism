from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_noise_multiplier
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.dp_utils import scatter_normalization
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp, train_with_dp_cmdp
from utils.dp_optimizer import  DPSGD_Optimizer
import torch

from algorithm.CMDP_Mechanism import hyper_para_setup

from train_and_validation.validation import validation


from data.util.sampling import get_data_loaders_possion
import os

def DPSGD_HF_CMDP(dataset_name,train_data, test_data, model, batch_size, hyper_para, momentum, epsilon_budget, C_t, sigma, use_scattering,input_norm, bn_noise_multiplier, num_groups, device, log_file):

    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)


    if use_scattering:
        scattering, K, _ = get_scatter_transform(dataset_name)
        scattering.to(device)
    else:
        scattering = None
        K = 3 if len(train_data.data.shape) == 4 else 1


    rdp_norm=0.
    if input_norm == "BN":
        # compute noisy data statistics or load from disk if pre-computed
        save_dir = f"bn_stats/{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                   scattering,
                                                   K,
                                                   device,
                                                   len(train_data),
                                                   len(train_data),
                                                   noise_multiplier=bn_noise_multiplier,
                                                   save_dir=save_dir)
        model = CNNS[dataset_name](K, input_norm="BN", bn_stats=bn_stats, size=None)

    else:
        model = CNNS[dataset_name](K, input_norm=input_norm, num_groups=num_groups, size=None)

    model.to(device)

    train_data_scattered = get_scattered_dataset(train_loader, scattering, device, len(train_data))
    test_loader = get_scattered_loader(test_loader, scattering, device)

    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size, microbatch_size=1, iterations=1)


    optimizer = DPSGD_Optimizer(
        l2_norm_clip=C_t,
        noise_multiplier=sigma,
        minibatch_size=batch_size,
        microbatch_size=1,
        params=model.parameters(),
        lr=hyper_para.lr,
        momentum=momentum,
    )

    iter = 1
    epsilon_list=[]
    test_loss_list=[]
    best_test_acc=0.

    RDP_con = 0.
    epsilon_total = epsilon_budget
    epsilon_used = hyper_para.epsilon
    N_epoch = 10
    lr_used = hyper_para.lr
    epsilon_con = 0.

    while epsilon_con < epsilon_total:

        train_dl = minibatch_loader(train_data_scattered)
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp_cmdp(model, train_dl, optimizer, epsilon_used, lr_used, device)

        test_loss, test_accuracy = validation(model, test_loader, device)

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

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}
