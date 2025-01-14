import argparse
import os
import pickle
from datetime import time, datetime

import pandas as pd
import torch

from algorithm.DPAGD import DPAGD
from algorithm.DPSGD import DPSGD
from algorithm.DPSGD_HF import DPSGD_HF
from algorithm.DPSGD_TS import DPSGD_TS
from algorithm.DPSUR import DPSUR
from algorithm.DPSGD_CMDP import DPSGD_CMDP
from algorithm.DPAGD_CMDP import DPAGD_CMDP
from algorithm.DPSGD_TS_CMDP import DPSGD_TS_CMDP
from algorithm.DPSGD_HF_CMDP import DPSGD_HF_CMDP
from algorithm.DPSUR_CMDP import DPSUR_CMDP
from algorithm.CMDP_Mechanism import hyper_para_setup

from data.util.get_data import get_data
from data.util.prepare_MIA_dataset import prepare_MIA_dataset
from membership_inference.Inference_attacks import Member_inference_attacks
from membership_inference.test_meminf import test_meminf

from model.get_model import get_model
from utils.dp_optimizer import  get_dp_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="DPSGD",choices=['DPSGD', 'DPSGD-TS', 'DPSGD-HF', 'DPSUR', 'DPAGD', 'DPSGD-CMDP', 'DPSGD-TS-CMDP', 'DPSGD-HF-CMDP', 'DPSUR-CMDP', 'DPAGD-CMDP'])
    parser.add_argument('--dataset_name', type=str, default="MNIST",choices=['MNIST', 'FMNIST', 'CIFAR-10', 'IMDB'])
    parser.add_argument('--lr', type=float, default=2)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--use_scattering', action="store_true")
    parser.add_argument('--input_norm', default=None, choices=["GroupNorm", "BN"])
    parser.add_argument('--bn_noise_multiplier', type=float, default=8)
    parser.add_argument('--num_groups', type=int, default=27)

    parser.add_argument('--sigma_t', type=float, default=1.23)
    parser.add_argument('--C_t', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=3.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--sigma_v', type=float, default=1.0)
    parser.add_argument('--C_v', type=float, default=0.001)
    parser.add_argument('--bs_valid', type=int, default=256)
    parser.add_argument('--beta', type=float, default=-1.0)
    parser.add_argument('--ini_alpha', type=float, default=10.0)

    parser.add_argument('--MIA', type=bool, default=False)

    parser.add_argument('--device', type=str, default='cpu',choices=['cpu', 'cuda'])

    parser.add_argument('--lr_T_gain', type=float, default=1.01)
    parser.add_argument('--lr_max', type=float, default=4.0)
    parser.add_argument('--lr_min', type=float, default=0.5)

    parser.add_argument('--epsilon_T_acc', type=float, default=90)
    parser.add_argument('--C_t_T_acc', type=float, default=90)



    args = parser.parse_args()

    algorithm=args.algorithm
    dataset_name=args.dataset_name
    lr=args.lr
    momentum=args.momentum

    use_scattering=args.use_scattering
    input_norm=args.input_norm
    bn_noise_multiplier=args.bn_noise_multiplier
    num_groups=args.num_groups

    sigma_t=args.sigma_t
    C_t=args.C_t
    epsilon=args.epsilon
    delta=args.delta
    batch_size=args.batch_size

    sigma_v=args.sigma_v
    bs_valid=args.bs_valid
    C_v=args.C_v
    beta=args.beta
    ini_alpha=args.ini_alpha

    MIA=args.MIA

    device=args.device

    lr_T_gain=args.lr_T_gain
    epsilon_T_acc=args.epsilon_T_acc
    lr_max=args.lr_max
    lr_min=args.lr_min

    hyper_para = hyper_para_setup(lr, lr_T_gain, lr_min, lr_max, epsilon_T_acc)

    if MIA:
        result_dir = os.getcwd() + f"/result/MIA/{algorithm}/{dataset_name}/{epsilon}/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        result_dir = os.getcwd() + f"/result/Without_MIA/{algorithm}/{dataset_name}/{epsilon}/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    # 创建日志文件
    log_file = os.path.join(result_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    if MIA:
        print("MIA is starting...")
        num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_MIA_dataset(dataset_name, algorithm,device)
        optimizer = get_dp_optimizer(dataset_name, algorithm, lr, momentum, C_t, sigma_t, batch_size, target_model, ini_alpha)

        if algorithm=='DPSGD':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD(target_train, target_test, target_model,optimizer, batch_size, epsilon, delta,sigma_t,device, log_file)
        elif algorithm == 'DPSGD-CMDP':
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD_CMDP(target_train, target_test, target_model, optimizer, hyper_para, batch_size, epsilon, device, log_file)
        elif algorithm == 'DPSGD-TS':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD_TS(target_train, target_test, target_model,optimizer, batch_size, epsilon, delta,sigma_t,device, log_file)
        elif algorithm == 'DPSGD-TS-CMDP':
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD_TS_CMDP(target_train, target_test, target_model, optimizer, hyper_para, batch_size, epsilon, device, log_file)
        elif algorithm=='DPAGD':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPAGD(target_train, target_test, target_model,optimizer, batch_size, epsilon, delta,sigma_t, C_v,sigma_v,device,log_file)
        elif algorithm=='DPAGD-CMDP':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPAGD_CMDP(target_train, target_test, target_model, optimizer, hyper_para, batch_size, epsilon, C_v, sigma_v, device, log_file)
        elif algorithm == "DPSUR":
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSUR(dataset_name,target_train, target_test, target_model, batch_size, lr, momentum, epsilon,delta, C_t,
                   sigma_t,use_scattering,input_norm,bn_noise_multiplier,num_groups,bs_valid,C_v,beta,sigma_v,MIA,device, log_file)
        elif algorithm == "DPSUR-CMDP":
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSUR_CMDP(dataset_name,target_train, target_test, target_model, batch_size, hyper_para, momentum, epsilon, C_t,
                   sigma_t,use_scattering,input_norm,bn_noise_multiplier,num_groups,bs_valid,C_v,beta,sigma_v,MIA, device, log_file)
        else:
            raise ValueError("this algorithm is not exist")

    else:

        train_data, test_data,dataset = get_data(dataset_name, augment=False)
        model=get_model(algorithm,dataset_name,device)
        optimizer = get_dp_optimizer(dataset_name, algorithm,lr, momentum, C_t, sigma_t, batch_size, model, ini_alpha)

        if algorithm=='DPSGD':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD(train_data, test_data, model,optimizer, batch_size, epsilon, delta,sigma_t,device,log_file)

        elif algorithm=='DPSGD-CMDP':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD_CMDP(train_data, test_data, model, optimizer, hyper_para, batch_size, epsilon, device, log_file)

        elif algorithm=='DPAGD':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPAGD(train_data, test_data, model,optimizer, batch_size, epsilon, delta,sigma_t, C_v,sigma_v,device,log_file)

        elif algorithm=='DPAGD-CMDP':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPAGD_CMDP(train_data, test_data, model, optimizer, hyper_para, batch_size, epsilon, C_v, sigma_v, device, log_file)

        elif algorithm == 'DPSGD-TS':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD_TS(train_data, test_data, model,optimizer, batch_size, epsilon, delta,sigma_t,device,log_file)

        elif algorithm=='DPSGD-TS-CMDP':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD_TS_CMDP(train_data, test_data, model, optimizer, hyper_para, batch_size, epsilon, device, log_file)

        elif algorithm == 'DPSGD-HF' and dataset_name !='IMDB':  #Not support IMDB
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD_HF(dataset_name, train_data, test_data, model, batch_size, lr, momentum, epsilon, delta,
                     C_t, sigma_t, use_scattering, input_norm, bn_noise_multiplier, num_groups, device,log_file)

        elif algorithm == 'DPSGD-HF-CMDP' and dataset_name !='IMDB':  #Not support IMDB
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD_HF_CMDP(dataset_name, train_data, test_data, model, batch_size, hyper_para, momentum, epsilon,
                     C_t, sigma_t, use_scattering, input_norm, bn_noise_multiplier, num_groups, device, log_file)

        elif algorithm == "DPSUR":
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSUR(dataset_name,train_data, test_data, model, batch_size, lr, momentum, epsilon,delta, C_t,
                   sigma_t,use_scattering,input_norm,bn_noise_multiplier,num_groups,bs_valid,C_v,beta,sigma_v,MIA,device,log_file)

        elif algorithm == "DPSUR-CMDP":
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSUR_CMDP(dataset_name,train_data, test_data, model, batch_size, hyper_para, momentum, epsilon, C_t,
                   sigma_t,use_scattering,input_norm,bn_noise_multiplier,num_groups,bs_valid,C_v,beta,sigma_v,MIA, device, log_file)

        else:
            raise ValueError("this algorithm is not exist")



    if MIA:
        File_Path_Csv = os.getcwd() + f"/result/MIA/{algorithm}/{dataset_name}/{epsilon}//"
        if not os.path.exists(File_Path_Csv):
            os.makedirs(File_Path_Csv)
        result_path = f'{File_Path_Csv}/{str(sigma_t)}_{str(lr)}_{str(batch_size)}_{str(sigma_v)}_{str(bs_valid)}.csv'
        pd.DataFrame([best_acc, int(best_iter), test_acc, int(last_iter)]).to_csv(result_path, index=False,header=False)
        target_model_path=f'{File_Path_Csv}/{str(sigma_t)}_{str(lr)}_{str(batch_size)}_{str(sigma_v)}_{str(bs_valid)}_model.pth'
        torch.save(trained_model.state_dict(), target_model_path)

        shadow_model_path=f'{File_Path_Csv}/{str(sigma_t)}_{str(lr)}_{str(batch_size)}_{str(sigma_v)}_{str(bs_valid)}_shadow_model.pth'
        attack_path=f'{File_Path_Csv}/{str(sigma_t)}_{str(lr)}_{str(batch_size)}_{str(sigma_v)}_{str(bs_valid)}_attack.pth'
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> membership inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        meminf_res_train0, meminf_res_test0,meminf_res_train3, meminf_res_test3, meminf_res_train2,meminf_res_test2 = test_meminf(device, num_classes, target_train, target_test, shadow_train, shadow_test,
                target_model, shadow_model,target_model_path,shadow_model_path,attack_path)
    else:
        File_Path_Csv = os.getcwd() + f"/result/Without_MIA/{algorithm}/{dataset_name}/{epsilon}//"
        if not os.path.exists(File_Path_Csv):
            os.makedirs(File_Path_Csv)
        result_path = f'{File_Path_Csv}/{str(sigma_t)}_{str(lr)}_{str(batch_size)}_{str(sigma_v)}_{str(bs_valid)}.csv'
        pd.DataFrame([best_acc, int(best_iter), test_acc, int(last_iter)]).to_csv(result_path, index=False,
                                                                                  header=False)
        torch.save(iter_list, f"{File_Path_Csv}/iterList.pth")


if __name__=="__main__":

    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    main()
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start time: ", start_time)
    print("end time: ", end_time)
