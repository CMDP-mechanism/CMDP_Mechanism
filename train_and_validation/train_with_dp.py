import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from privacy_analysis.RDP.compute_rdp import compute_rdp
from train_and_validation.validation import validation, validation_per_sample
import numpy as np

from utils.NoisyMax import NoisyMax
import math
from privacy_analysis.RDP.rdp_convert_dp import compute_eps

def train_with_dp(model, train_loader, optimizer,device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  #改为负数似然损失函数了，后面记得要改回来

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp()


    return train_loss, train_acc

def train_with_dp_cmdp(model, train_loader, optimizer, epsilon, lr_used, device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp_cmdp(epsilon, lr_used)


    return train_loss, train_acc

def train_with_dp_sgd_comp2(model, train_loader, optimizer, epsilon, lr_used, device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  #改为负数似然损失函数了，后面记得要改回来

            loss.backward()
            optimizer.microbatch_step()

        # 在执行参数更新之前打印学习率
        current_lr = optimizer.param_groups[0]['lr']  # 假设所有参数组使用相同的学习率
        # print(f"Current learning rate: {current_lr}")

        optimizer.step_dp_cmdp(epsilon, lr_used)

    return train_loss, train_acc

def apply_gradients(model, gradients, lr):
    # 模拟更新参数
    with torch.no_grad():
        for param, grad in zip(model.parameters(), gradients):
            param -= lr * grad
def select_learning_rate(model, original_gradients, train_loader, device, learning_rates, C_v, sigma_v, n):
    losses = []
    for lr in learning_rates:
        # 模拟更新参数
        apply_gradients(model, original_gradients, lr)
        test_loss = validation_per_sample(model, train_loader, device, C_v)
        losses.append(test_loss)
        # 还原参数
        apply_gradients(model, original_gradients, -lr)

    # 使用NoisyMax选择最小损失的学习率
    min_index = NoisyMax(losses, sigma_v, C_v, n, device)
    return learning_rates[min_index], min_index

def train_with_dp_agd_comp(model, train_loader, optimizer, ini_epsilon, lr_used, hyper_para, C_v, sigma_v, q, device):
    # print("train with adg2")
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    C_t = optimizer.l2_norm_clip
    epsilon_used = ini_epsilon

    RDP = 0


    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        original_gradients = [param.grad.clone() for param in model.parameters()]
        optimizer.step_agd_no_update_grad(epsilon_used)

        # 学习率选择
        learning_rate = np.linspace(hyper_para.lr_min, hyper_para.lr_max, 10)
        min_lr, min_index = select_learning_rate(model, original_gradients, train_loader, device, learning_rate, C_v, sigma_v, len(target))

        if min_index > 0:
            gamma = 1.01
            epsilon_used = min(epsilon_used * gamma, ini_epsilon * 1.05)

            RDP = RDP + optimizer.epsilon_R_subsampling(q)

        # 恢复原始梯度并更新参数
        for param, grad in zip(model.parameters(), original_gradients):
            param.grad = grad

        lr_used = min_lr

        train_loss, train_acc = train_with_dp_sgd_comp2(model, train_loader, optimizer, epsilon_used, lr_used, device)

        # train_loss, train_acc = train_with_dp_sgd2(model, train_loader, optimizer, device)


    return train_loss, train_acc, RDP, lr_used

def train_with_dp_agd(model, train_loader, optimizer,C_t,sigma_t,C_v,sigma_v,device,batch_size,num):
    model.train()
    train_loss = 0.0
    train_acc=0.
    noise_multiplier = sigma_t
    RDP=0.
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]

    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()


        # 获取原参数和裁剪的梯度值,这个是为了后面可能重加噪用的
        model_parameters_clipped = model.parameters()   #获取原参数
        gradients_clipped = [param.grad.clone() for param in model_parameters_clipped]


        optimizer.step_dp_agd()   #只是进行了梯度加噪，没有进行梯度下降
        # 获取原参数和裁剪加噪平均后的梯度值
        model_parameters = model.parameters()
        gradients = [param.grad.clone() for param in model_parameters]

        model_parameters_dict = model.state_dict()


        learning_rate = np.linspace(0, 1, 20 + 1)  # 学习率从0-2.0分成20份

        min_index=0

        while min_index == 0:
            loss = []
            for i, lr in enumerate(learning_rate):
                # 更新参数
                with torch.no_grad():

                    for param, gradient in zip(model_parameters_dict.values(), gradients):
                        param -= lr * gradient

                    model.load_state_dict(model_parameters_dict)
                    test_loss = validation_per_sample(model, train_loader, device,C_v)
                    loss.append(test_loss)

            # 找最小值
            min_index = NoisyMax(loss, sigma_v, C_v, len(target))


            if min_index > 0:
                # 拿到使得这次loss最小的梯度值
                lr = learning_rate[min_index]
                with torch.no_grad():
                    for param, gradient in zip(model_parameters_dict.values(), gradients):
                        param -= lr * gradient
                model.load_state_dict(model_parameters_dict)
                RDP = RDP + compute_rdp(batch_size / num, sigma_v, 1, orders)  # 这个是进行loss计算，选择最佳学习率的RDP

            else:
                #如果是0最佳的，那么需要进行隐私预算加大，即多分配隐私预算，然后sigma变小
                # 对g进行重加噪，用小的sigma进行重加噪，然后隐私资源消耗
                gamma=0.99
                noise_multiplier=sigma_t*gamma
                print("noise_multiplier:",noise_multiplier)
                with torch.no_grad():
                    for gradient1, gradient2 in zip(gradients_clipped, gradients):

                         gradient1+=C_t * noise_multiplier * torch.randn_like(gradient1)
                         gradient1/=len(target)
                         gradient2=(noise_multiplier*gradient1+(sigma_t-noise_multiplier)*gradient2)/(sigma_t)

                #更新model里的

                RDP = RDP + compute_rdp(batch_size / num, noise_multiplier, 1,orders)  # 这个是重加噪的RDP


    return train_loss, train_acc,noise_multiplier,RDP

def train_with_dp_agd2(model, train_loader, optimizer,C_t,sigma_t,C_v,sigma_v,device,batch_size,num):
    model.train()
    train_loss = 0.0
    train_acc=0.
    noise_multiplier = sigma_t
    RDP=0.
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]

    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()


        # 获取原参数和裁剪的梯度值,这个是为了后面可能重加噪用的
        model_parameters_clipped = model.parameters()   #获取原参数
        gradients_clipped = [param.grad.clone() for param in model_parameters_clipped]


        optimizer.step_dp_agd()   #只是进行了梯度加噪，没有进行梯度下降
        # 获取原参数和裁剪加噪平均后的梯度值
        model_parameters = model.parameters()
        gradients = [param.grad.clone() for param in model_parameters]

        model_parameters_dict = model.state_dict()


        #learning_rate = np.linspace(0, 5, 20 + 1)  # 学习率从0-2.0分成20份
        learning_rate = np.linspace(0., 0.1, 5 + 1)  # 学习率从0-0.1分成10份

        min_index=0
        loss = []

        for i, lr in enumerate(learning_rate):
            # 更新参数
            with torch.no_grad():

                for param, gradient in zip(model_parameters_dict.values(), gradients):
                    param -= lr * gradient

                model.load_state_dict(model_parameters_dict)
                test_loss = validation_per_sample(model, train_loader, device,C_v)
                loss.append(test_loss)

        # 找最小值
        min_index = NoisyMax(loss, sigma_v, C_v, len(target))


        if min_index > 0:
            # 拿到使得这次loss最小的梯度值
            lr = learning_rate[min_index]
            with torch.no_grad():
                for param, gradient in zip(model_parameters_dict.values(), gradients):
                    param -= lr * gradient
            model.load_state_dict(model_parameters_dict)
            RDP = RDP + compute_rdp(batch_size / num, sigma_v, 1, orders)  # 这个是进行loss计算，选择最佳学习率的RDP

        else:
            #如果是0最佳的，那么需要进行隐私预算加大，即多分配隐私预算，然后sigma变小
            # 对g进行重加噪，用小的sigma进行重加噪，然后隐私资源消耗
            noise_multiplier=sigma_t*0.9998
            print("noise_multiplier:",noise_multiplier)
            with torch.no_grad():
                for gradient1, gradient2 in zip(gradients_clipped, gradients):

                     gradient1+=C_t * noise_multiplier * torch.randn_like(gradient1)
                     gradient1/=len(target)
                     gradient2=(noise_multiplier*gradient1+sigma_t*gradient2)/(noise_multiplier+sigma_t)

            RDP = RDP + compute_rdp(batch_size / num, sigma_t, 1,orders)  # 这个是重加噪的RDP


    return train_loss, train_acc,noise_multiplier,RDP
