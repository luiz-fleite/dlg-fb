import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import PIL.Image as Image
import csv


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def main():
    # Choose dataset here
    dataset = "CIFAR100"
    root_path = "."

    # on linux evironment
    data_path = os.path.join(root_path, "datasets/dlg-FB-tests")
    save_path = os.path.join(root_path, f"results/DLG-FB_{dataset}")
    
    lr = 1.0

    #CONVERGENCE_LOSS = 0.0000005 # trying to force more accuracy
    CONVERGENCE_LOSS = 0.000001 # first converngence loss
    #CONVERGENCE_LOSS = 0.00001 # relaxing 


    num_dummy = 1
    TOTAL_ITERATIONS = 200
    TOTAL_EXP = 20

    # Tracking time
    date_print_format = "[%Y/%m/%d %H:%M:%S]"
    date_file_name_format = "[%Y-%m-%d_%H-%M-%S]"
    INITIAL_TIME = datetime.datetime.now()
    INITIAL_TIME_STR = INITIAL_TIME.strftime(date_file_name_format)

    # creating csv file
    # defining header
    header = ["img_idx", "method", "initializer", "exp", "iters", "gt_label", "dummy_label", "pred_label", "converged", "loss", "mse"]

    # creating diretories
    csv_save_path = os.path.join(root_path, f"metrics/tests_{dataset}")
    if not os.path.exists("metrics"):
        os.mkdir("metrics")
    if not os.path.exists(csv_save_path):
        os.mkdir(csv_save_path)

    # writing csv file header
    with open(f"{csv_save_path}/metrics_{dataset}_at_{INITIAL_TIME_STR}.csv", "a") as f:
        write = csv.writer(f)
        write.writerow(header)

    # creating device agnostic code
    #use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = 'cuda' if use_cuda else 'cpu'

    print("===============================")
    print("Using cuda: ", use_cuda)
    print("===============================")
    print("Device: ", device)
    print("===============================")

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)



    ''' load data '''
    if dataset == "MNIST":
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == "CIFAR100":
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)


    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, '../datasets/lfw')
        dst = lfw_dataset(lfw_path, shape_img)
    
    elif dataset == 'my_test':
        shape_img = (1, 1)
        num_classes = 2
        channel = 1
        hidden = 2
        dst = [(torch.tensor([0.]), 0), (torch.tensor([1.]), 1)]

    else:
        exit('unknown dataset')

    # defining the methods and initializers
    methods = ["DLG", "iDLG"]
    #methods = ["DLG"]
    #methods = ["iDLG"]

    #initializers = ["random", "FB-NF"]
    initializers = ["random", "FB-NF", "FB"]
    #initializers_1 = ["random", "FB"]

    # creating dummy data records for FB initializer
    dummy_data = {}
    dummy_data = {method:{initializer:{"old": None, "new": None} for initializer in initializers} for method in methods}
    # creating exception for FB failures
    for method in methods:
        dummy_data[method]["FB"]["failed"] = False

    # tracking metrics
    metrics = {}
    metrics = {method:{initializer:{"losses_history": [], "mses_history": [], "iterations_history": [], "convergences": 0, "failures": 0, "gt_label_history": [],"dummy_logit_history": [],  "pred_logit_history": []} for initializer in initializers} for method in methods}


    # running DLG-FB combinations
    for exp in range(TOTAL_EXP):
        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print(f"running {exp}|{TOTAL_EXP} experiment")
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        for method in methods:
            print(f"{method}, Try to generate {num_dummy} images")

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    # Teste com imagem uniforme
                    #tmp_datum = tmp_datum.fill_(0.)

                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)


            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # creating FB tracker in broader scope
            fb_started = False

            # generate dummy data and label
            # exploring different initializers
            for initializer in initializers:

                if initializer == "random":
                    dummy_data[method]["random"]["new"] = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                    
                    if exp == 0:
                        # starting with random dummy_data for FB-NF initializer
                        dummy_data[method]["FB-NF"]["old"] = dummy_data[method]["random"]["new"].clone().detach().requires_grad_(True)
                    

                elif initializer == "FB-NF":
                    
                    # Chosing an ALPHA value to blend the old and new dummy_data
                    ALPHA = 0.5

                    # blending the old and new dummy_data
                    dummy_data[method]["FB-NF"]["old"] = torch.mul(dummy_data[method]["FB-NF"]["old"], ALPHA).clone().detach().to(device).requires_grad_(True)
                    dummy_data[method]["FB-NF"]["new"] = torch.mul(dummy_data[method]["FB-NF"]["new"], 1-ALPHA).clone().detach().to(device).requires_grad_(True)
                    dummy_data[method]["FB-NF"]["new"] = torch.add(dummy_data[method]["FB-NF"]["old"], dummy_data[method]["FB-NF"]["new"]).clone().detach().to(device).requires_grad_(True)

                    # updates the FB-NF dummy_data_record (old) with the new dummy_data
                    dummy_data[method]["FB-NF"]["old"] = dummy_data[method]["FB-NF"]["new"].clone().detach().to(device).requires_grad_(True)

                elif initializer == "FB":
                        if metrics[method]["random"]["convergences"] < 3:
                            continue

                        # FB starts if random initializer has converged twice + skips 1 image = 3 experiments
                        fb_started = True
                        # just debugging
                        print("FB started!")
                        
                        # Chosing an ALPHA value to blend the old and new dummy_data
                        ALPHA = 0.5
                        
                        if dummy_data[method]["FB"]["failed"]:
                            # if FB has failed, it must restart from last checkpoint
                            dummy_data[method]["FB"]["new"] = dummy_data[method]["FB"]["old"].clone().detach().to(device).requires_grad_(True)
                            # updates variable
                            dummy_data[method]["FB"]["failed"] = False
                        else:
                            # blending the old and new dummy_data
                            dummy_data[method]["FB"]["old"] = torch.mul(dummy_data[method]["FB"]["old"], ALPHA).clone().detach().to(device).requires_grad_(True)
                            dummy_data[method]["FB"]["new"] = torch.mul(dummy_data[method]["FB"]["new"], 1-ALPHA).clone().detach().to(device).requires_grad_(True)
                            dummy_data[method]["FB"]["new"] = torch.add(dummy_data[method]["FB"]["old"], dummy_data[method]["FB"]["new"]).clone().detach().to(device).requires_grad_(True)
        
                            # updates the FB dummy_data_record (old) with the new dummy_data
                            dummy_data[method]["FB"]["old"] = dummy_data[method]["FB"]["new"].clone().detach().to(device).requires_grad_(True)
                
                # dummy_label independent of initializer (by now)
                dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                if method == "DLG":
                    optimizer = torch.optim.LBFGS([dummy_data[method][initializer]["new"], dummy_label], lr=lr)
                elif method == "iDLG":
                    optimizer = torch.optim.LBFGS([dummy_data[method][initializer]["new"], ], lr=lr)

                    # predict the ground-truth label
                    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

                    # saving the label prediction
                    metrics[method][initializer]["dummy_logit_history"].append(label_pred.clone().detach())

                # tracking metrics
                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []
                pred_logit_history = []
                dummy_logit_history = []

                # reseting convergence tracker
                converged = False

                print('lr =', lr)
                for iters in range(TOTAL_ITERATIONS):

                    def plot_iters(is_first=False, is_last=False):
                        # print metrics
                        current_time = datetime.datetime.now().strftime(date_print_format)

                        if is_first:
                            print(f"{current_time} {iters} initializer: {initializer}")
                        else:
                            print(f"{current_time} {iters} initializer: {initializer}, loss = {current_loss:.8f}, mse = {mses[-1]:.8f}")

                        history.append([tp(dummy_data[method][initializer]["new"][imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        # plot dummy_data evolution
                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            plt.imshow(tp(gt_data[imidx].cpu()))
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)
                                plt.imshow(history[i][imidx])
                                if not is_last:
                                    plt.title(f"iter={history_iters[i]}")
                                elif is_last and converged:
                                    plt.title(f"iter={history_iters[i]}\nConverged!")
                                elif is_last and not converged:
                                    plt.title(f"iter={history_iters[i]}\nFail")
                                plt.axis('off')
                            
                            # saving the images with formated names
                            plt.savefig(f"{save_path}/exp_{exp}_{method}_on_img[{imidx_list[-1]}]_{initializer}.png")                               
                            plt.close()

                    # before starting iterations we show the initial state of dummy_data
                    # because its important for FB visualization
                    if iters == 0:
                        plot_iters(is_first=True)
                        # zero iteration is reserved for first visualization
                        # so we skip the rest of the loop
                        continue

                    def closure():
                        optimizer.zero_grad()
                        pred = net(dummy_data[method][initializer]["new"])
                        if method == "DLG":
                            dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == "iDLG":
                            dummy_loss = criterion(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                    optimizer.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data[method][initializer]["new"] - gt_data)**2).item())

                    # tracking dummy_label and pred_label
                    if method == "DLG":
                        dummy_logit_history.append(dummy_label.detach().clone().cpu().data.numpy())


                    pred_logit_history.append(net(dummy_data[method][initializer]["new"]).cpu().data.numpy())


                    if iters % int(TOTAL_ITERATIONS / 30) == 0 or iters in [0, 1]:
                    #if iters % 5 == 0 or iters in [0, 1]:
                        # save dummy_data evolution at some determined step
                        plot_iters()
                        

                    if current_loss < CONVERGENCE_LOSS: # converge
                        converged = True
                        plot_iters(is_last=True)
                        print("converged!")

                        # tracking metrics
                        metrics[method][initializer]["convergences"] += 1

                        # the FB dirty initializer always updates the "new" dummy_data
                        # and at first iteration must get it from "random" initializer
                        if exp == 0:
                            dummy_data[method]["FB-NF"]["new"] = dummy_data[method]["random"]["new"].clone().detach().requires_grad_(True)

                        # the following block is for FB clean initializer
                        if initializer == "random":
                            if metrics[method]["random"]["convergences"] == 1:
                                # "old" dummy_data needs to be updated first
                                dummy_data[method]["FB"]["old"] = dummy_data[method]["random"]["new"].clone().detach().requires_grad_(True)
                            elif metrics[method]["random"]["convergences"] == 2:
                                # then updates the "new" dummy_data
                                dummy_data[method]["FB"]["new"] = dummy_data[method]["random"]["new"].clone().detach().requires_grad_(True)

                        break

                    # Trying to make a complex rule to know if not converged
                    if (iters == TOTAL_ITERATIONS - 1) or (iters > 20 and ((losses[-1] > 80 and (losses[-1] == losses[-2] == losses[-3] == losses[-4] == losses[-5])) or (mses[-1] > 180 and (mses[-1] == mses[-2] == mses[-3] == mses[-4] == mses[-5])))):
                        converged = False
                        plot_iters(is_last=True)
                        print("not converged!")

                        # tracking metrics
                        metrics[method][initializer]["failures"] +=1

                        # the FB dirty initializer always updates the "new" dummy_data
                        # and at first iteration must get it from "random" initializer
                        if exp == 0:
                            dummy_data[method]["FB-NF"]["new"] = dummy_data[method]["random"]["new"].clone().detach().requires_grad_(True)
                        
                        # fed back must skip if not converged, because it is clean
                        if initializer == "FB":
                            dummy_data[method]["FB"]["failed"] = True

                        break
                
                # tracking metrics 
                metrics[method][initializer]["losses_history"].append(losses)
                metrics[method][initializer]["mses_history"].append(mses)
                metrics[method][initializer]["iterations_history"].append(train_iters[-1])
                metrics[method][initializer]["gt_label_history"].append(gt_label[0].detach().cpu().tolist())
                metrics[method][initializer]["pred_logit_history"].append(pred_logit_history)
                if method == "DLG":
                    metrics[method][initializer]["dummy_logit_history"].append(dummy_logit_history)

                # simplifying some metrics
                if method == "DLG":
                    dummy_label_metric = torch.argmax(torch.from_numpy(np.array(metrics[method][initializer]["dummy_logit_history"][-1][-1]))).tolist()
                elif method == "iDLG":
                    dummy_label_metric = metrics[method][initializer]["dummy_logit_history"][-1].tolist()[0]
                pred_label_metric = torch.argmax(torch.from_numpy(np.array(metrics[method][initializer]["pred_logit_history"][-1][-1]))).tolist()

                # Saving to csv file
                with open(f"{csv_save_path}/metrics_{dataset}_at_{INITIAL_TIME_STR}.csv", "a") as f:
                    write = csv.writer(f)
                    write.writerow([ imidx_list[-1], method, initializer, exp, iters, gt_label[0].tolist(), dummy_label_metric, pred_label_metric, converged, current_loss, mses[-1]])
                    
        
                # skips FB initializer if it has not started
                if initializer == "FB" and not fb_started:
                    continue
                def print_metrics():
                    # printing all final results
                    print("============metrics============")
                    print("imidx_list: ", imidx_list[-1])
                    print("method: ", method, ', initializer:', initializer)
                    print("loss: ", metrics[method][initializer]["losses_history"][-1][-1])
                    print("mse: ", metrics[method][initializer]["mses_history"][-1][-1])
                    print("gt_label: ", metrics[method][initializer]["gt_label_history"][-1])
                    print("dummy_label: ", dummy_label_metric)
                    print("pred_label: ", pred_label_metric)
                    print("Total convergence: ", metrics[method][initializer]["convergences"])
                    print("Total failures: ", metrics[method][initializer]["failures"])
                    print("Initial time: ", INITIAL_TIME.strftime(date_print_format))
                    print("Final time: ", datetime.datetime.now().strftime(date_print_format))
                    print("===============================\n\n")

                print_metrics()


if __name__ == '__main__':
    main()


