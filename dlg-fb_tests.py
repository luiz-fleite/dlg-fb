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

    # Tracking time
    date_print_format = "[%Y/%m/%d %H:%M:%S]"
    date_file_name_format = "[%Y-%m-%d_%H-%M-%S]"
    INITIAL_TIME = datetime.datetime.now()
    INITIAL_TIME_STR = INITIAL_TIME.strftime(date_file_name_format)

    # on linux evironment 
    # defining directory structure for datasets, outputs and metrics
    data_path = os.path.join(root_path, "datasets/dlg-fb-tests")
    save_path = os.path.join(root_path, f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}/visualizations-{dataset}-{INITIAL_TIME_STR}")
    csv_save_path = os.path.join(root_path, f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}/metrics-csv")
    
    # printing paths
    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)
    print(dataset, 'csv_save_path:', csv_save_path)

    # creating directory structure
    if not os.path.exists('tests-outputs'):
        os.mkdir('tests-outputs')
    if not os.path.exists('tests-outputs/dlg-fb-outputs'):
        os.mkdir('tests-outputs/dlg-fb-outputs')
    if not os.path.exists(f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}"):
        os.mkdir(f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}")
    if not os.path.exists(f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}/visualizations-{dataset}-{INITIAL_TIME_STR}"):
        os.mkdir(f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}/visualizations-{dataset}-{INITIAL_TIME_STR}")
    if not os.path.exists(f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}/metrics-csv"):
        os.mkdir(f"tests-outputs/dlg-fb-outputs/dlg-fb-output-{INITIAL_TIME_STR}/metrics-csv")


    # Setting random seed for reproducibility 
    random_seed = 42 
    print("random_seed: ", random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False



    # defining lbfgs optimizer parameters 
    lr = 1.0

    # imagens com até 0.03 de loss devem ser consideradas vazamento de privacidade 
    # e talvez dê pra usar no blend no MNIST 
    # PORÉM para o CIFAR100 o valor é 0.09 
    #CONVERGENCE_LOSS = 0.0000005 # trying to force more accuracy
    CONVERGENCE_LOSS = 0.000001 # first converngence loss
    #CONVERGENCE_LOSS = 0.00001 # relaxing 
    print("CONVERGENCE_LOSS: ", CONVERGENCE_LOSS) 

    #CONVERGENCE_MSE = 0.0001

    num_dummy = 1
    TOTAL_ITERATIONS = 300 + 1
    TOTAL_EXP = 1200 

    # writing test's hiperparameter csv file 
    header_csv_hiperparameters_test = ["INITIAL_TIME", "root_path", "data_path", "save_path", "csv_save_path", "dataset", "num_dummy", "TOTAL_ITERATIONS", "TOTAL_EXP", "lr", "CONVERGENCE_LOSS"]
    with open(f"{csv_save_path}/metrics_hyperparameters_{dataset}_at_{INITIAL_TIME_STR}.csv", "a") as f:
        write = csv.writer(f)
        write.writerow(header_csv_hiperparameters_test)
        write.writerow([INITIAL_TIME.strftime(date_print_format), root_path, data_path, save_path, csv_save_path, dataset, num_dummy, TOTAL_ITERATIONS, TOTAL_EXP, lr, CONVERGENCE_LOSS])


    # defining the methods, initializers and defenses 
    methods = ["DLG", "iDLG"]
    #methods = ["DLG"]
    #methods = ["iDLG"]

    #initializers = ["random", "FB-NF"]
    #initializers = ["random", "FB-NF", "FB"]
    initializers = ["random", "FB"] 


    #defenses = ["None", "Gaussian Noise", "Laplacian Noise"] 
    #defenses = ["None", "Gaussian Noise"] 
    defenses = ["None"] 
    # magnitude is sensitivity / epsilon 
    # standard sensitivity is 1 and the epsilons are: 10, 100, 1000, 10000 
    #magnitudes = ["None", 0.1, 0.01, 0.001, 0.0001] 
    #magnitudes = ["None", 0.0001]
    magnitudes = ["None"]

    # creating dummy data records for FB initializer
    dummy_data = {}
    dummy_data = {method:{initializer:{defense:{magnitude:{"old": None, "new": None} for magnitude in magnitudes} for defense in defenses} for initializer in initializers} for method in methods}
    # creating exception for FB failures
    for method in methods:
        for defense in defenses:
            for magnitude in magnitudes:
                dummy_data[method]["FB"][defense][magnitude]["failed"] = False 
                dummy_data[method]["FB"][defense][magnitude]["started"] = False 


    # tracking metrics
    metrics = {}
    metrics = {method:{initializer:{defense:{magnitude:{"losses_history": [], "mses_history": [], "iterations_history": [], "iter_privacy_leaked": -1, "convergences": 0, "failures": 0, "gt_label_history": [],"dummy_logit_history": [],  "pred_logit_history": []} for magnitude in magnitudes} for defense in defenses} for initializer in initializers} for method in methods}


    # creating csv file
    # defining header
    header = ["img_idx", "method", "initializer", "defense", "magnitude", "exp", "iters", "iter_privacy_leaked", "gt_label", "dummy_label", "pred_label", "converged", "loss", "mse"]

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


            # generate dummy data and label
            # exploring different initializers 
            for initializer in initializers:
                for defense in defenses:
                    for magnitude in magnitudes: 
                        
                        # Skpping invalid combinations
                        if defense == "None" and magnitude != "None":
                            continue
                        if defense != "None" and magnitude == "None":
                            continue

                        # skips FB initializer if it has not started
                        if initializer == "FB" and metrics[method]["random"][defense][magnitude]["convergences"] < 3:
                            continue

                        if initializer == "random":
                            dummy_data[method]["random"][defense][magnitude]["new"] = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                            
                            if exp == 0 and "FB-NF" in initializers:
                                # starting with random dummy_data for FB-NF initializer
                                dummy_data[method]["FB-NF"][defense][magnitude]["old"] = dummy_data[method]["random"][defense][magnitude]["new"].clone().detach().requires_grad_(True)
                            

                        elif initializer == "FB-NF":
                            
                            # Chosing an ALPHA value to blend the old and new dummy_data
                            ALPHA = 0.5

                            # blending the old and new dummy_data
                            dummy_data[method]["FB-NF"][defense][magnitude]["old"] = torch.mul(dummy_data[method]["FB-NF"][defense][magnitude]["old"], ALPHA).clone().detach().to(device).requires_grad_(True)
                            dummy_data[method]["FB-NF"][defense][magnitude]["new"] = torch.mul(dummy_data[method]["FB-NF"][defense][magnitude]["new"], 1-ALPHA).clone().detach().to(device).requires_grad_(True)
                            dummy_data[method]["FB-NF"][defense][magnitude]["new"] = torch.add(dummy_data[method]["FB-NF"][defense][magnitude]["old"], dummy_data[method]["FB-NF"][defense][magnitude]["new"]).clone().detach().to(device).requires_grad_(True)

                            # updates the FB-NF dummy_data_record (old) with the new dummy_data
                            dummy_data[method]["FB-NF"][defense][magnitude]["old"] = dummy_data[method]["FB-NF"][defense][magnitude]["new"].clone().detach().to(device).requires_grad_(True)

                        elif initializer == "FB":
                                if metrics[method]["random"][defense][magnitude]["convergences"] < 3:
                                    continue

                                # FB starts if random initializer has converged twice + skips 1 image = 3 experiments
                                dummy_data[method]["FB"][defense][magnitude]["started"] = True
                                # just debugging
                                print("FB started!")
                                
                                # Chosing an ALPHA value to blend the old and new dummy_data
                                ALPHA = 0.5
                                
                                if dummy_data[method]["FB"][defense][magnitude]["failed"]:
                                    # if FB has failed, it must restart from last checkpoint
                                    dummy_data[method]["FB"][defense][magnitude]["new"] = dummy_data[method]["FB"][defense][magnitude]["old"].clone().detach().to(device).requires_grad_(True)
                                    # updates variable
                                    dummy_data[method]["FB"][defense][magnitude]["failed"] = False
                                else:
                                    # blending the old and new dummy_data
                                    dummy_data[method]["FB"][defense][magnitude]["old"] = torch.mul(dummy_data[method]["FB"][defense][magnitude]["old"], ALPHA).clone().detach().to(device).requires_grad_(True)
                                    dummy_data[method]["FB"][defense][magnitude]["new"] = torch.mul(dummy_data[method]["FB"][defense][magnitude]["new"], 1-ALPHA).clone().detach().to(device).requires_grad_(True)
                                    dummy_data[method]["FB"][defense][magnitude]["new"] = torch.add(dummy_data[method]["FB"][defense][magnitude]["old"], dummy_data[method]["FB"][defense][magnitude]["new"]).clone().detach().to(device).requires_grad_(True)
                
                                    # updates the FB dummy_data_record (old) with the new dummy_data
                                    dummy_data[method]["FB"][defense][magnitude]["old"] = dummy_data[method]["FB"][defense][magnitude]["new"].clone().detach().to(device).requires_grad_(True)
                        
                        # dummy_label independent of initializer (by now)
                        dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                        if method == "DLG":
                            optimizer = torch.optim.LBFGS([dummy_data[method][initializer][defense][magnitude]["new"], dummy_label], lr=lr)
                        elif method == "iDLG":
                            optimizer = torch.optim.LBFGS([dummy_data[method][initializer][defense][magnitude]["new"], ], lr=lr)

                            # predict the ground-truth label
                            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

                            # saving the label prediction
                            metrics[method][initializer][defense][magnitude]["dummy_logit_history"] = label_pred.clone().detach()

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
                        # variable to track in which iteration privacy has humanly leaked
                        iter_privacy_leaked = -1

                        print('lr =', lr)

                        for iters in range(TOTAL_ITERATIONS):

                            def plot_iters(is_first=False, is_last=False):
                                # print metrics
                                current_time = datetime.datetime.now().strftime(date_print_format)

                                if is_first:
                                    print(f"{current_time} {iters} initializer: {initializer}")
                                else:
                                    print(f"{current_time} {iters} initializer: {initializer}, defense: {defense}, magnitude: {magnitude}, loss = {current_loss:.8f}, mse = {mses[-1]:.8f}")

                                history.append([tp(dummy_data[method][initializer][defense][magnitude]["new"][imidx].cpu()) for imidx in range(num_dummy)])
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
                                    plt.savefig(f"{save_path}/exp_{exp}_{method}_on_img[{imidx_list[-1]}]_{initializer}_{defense}_{magnitude}.png")                               
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
                                pred = net(dummy_data[method][initializer][defense][magnitude]["new"])
                                if method == "DLG":
                                    dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                                    # dummy_loss = criterion(pred, gt_label)
                                elif method == "iDLG":
                                    dummy_loss = criterion(pred, label_pred)

                                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                                # if defesence is None just keep the gradients as they are 
                                # else apply the noise to the gradients
                                if defense != "None":
                                    for grad in dummy_dy_dx:
                                        if defense == "Gaussian Noise":
                                            noise = torch.normal(mean=0, std=magnitude, size=grad.size()).to(device)
                                            grad += noise
                                        elif defense == "Laplacian Noise":
                                            noise = torch.distributions.laplace.Laplace(0, magnitude).sample(grad.size()).to(device)
                                            grad += noise
                                        else:
                                            raise ValueError("Unknown defense") 

                                grad_diff = 0
                                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                                    grad_diff += ((gx - gy) ** 2).sum()
                                grad_diff.backward()
                                return grad_diff

                            optimizer.step(closure)
                            current_loss = closure().item()
                            train_iters.append(iters)
                            losses.append(current_loss)
                            mses.append(torch.mean((dummy_data[method][initializer][defense][magnitude]["new"] - gt_data)**2).item())

                            # tracking dummy_label and pred_label
                            if method == "DLG":
                                dummy_logit_history = dummy_label.detach().clone().cpu().data.numpy()


                            pred_logit_history = net(dummy_data[method][initializer][defense][magnitude]["new"]).cpu().data.numpy()


                            if iters % int(TOTAL_ITERATIONS / 30) == 0 or iters in [0, 1]:
                            #if iters % 5 == 0 or iters in [0, 1]:
                                # save dummy_data evolution at some determined step
                                plot_iters()
                                
                            # detecting convergence humanly perceptible 
                            if not converged and ((dataset == "MNIST" and current_loss <= 0.03) or (dataset == "CIFAR100" and current_loss <= 0.09)):
                                converged = True 
                                iter_privacy_leaked = iters 
                                print(f"Privacy leaked at iteration: {iter_privacy_leaked}!")
                                # tracking metrics
                                metrics[method][initializer][defense][magnitude]["convergences"] += 1


                            # accurate convergence 
                            if current_loss <= CONVERGENCE_LOSS or (converged and iters == TOTAL_ITERATIONS -1): # converge
                                converged = True
                                plot_iters(is_last=True)
                                print("converged!")



                                # the FB dirty initializer always updates the "new" dummy_data
                                # and at first iteration must get it from "random" initializer
                                if exp == 0 and "FB-NF" in initializers:
                                    dummy_data[method]["FB-NF"][defense][magnitude]["new"] = dummy_data[method]["random"][defense][magnitude]["new"].clone().detach().requires_grad_(True)

                                # the following block is for FB clean initializer
                                if initializer == "random":
                                    if metrics[method]["random"][defense][magnitude]["convergences"] == 1:
                                        # "old" dummy_data needs to be updated first
                                        dummy_data[method]["FB"][defense][magnitude]["old"] = dummy_data[method]["random"][defense][magnitude]["new"].clone().detach().requires_grad_(True)
                                    elif metrics[method]["random"][defense][magnitude]["convergences"] == 2:
                                        # then updates the "new" dummy_data
                                        dummy_data[method]["FB"][defense][magnitude]["new"] = dummy_data[method]["random"][defense][magnitude]["new"].clone().detach().requires_grad_(True)

                                break

                            # Trying to make a complex rule to know if not converged 
                            # and cut some iterations trying to not stop a possible success 
                            if (iters == TOTAL_ITERATIONS - 1) or (iters > 20 and ((losses[-1] > 80 and (losses[-1] >= losses[-2] >= losses[-3] >= losses[-4] >= losses[-5])) or (mses[-1] > 180 and (mses[-1] >= mses[-2] >= mses[-3] >= mses[-4] >= mses[-5]))) or (iters > 80 and losses[-1] > 300 and losses[-2] > 300 and losses[-3] > 300 and losses[-4] > 300 and losses[-5] > 300 and losses[-6] > 300 and losses[-7] > 300 and losses[-8] > 300 and losses[-9] > 300 and losses[-10] > 300) or (iters > 80 and mses[-1] > 300 and mses[-2] > 300 and mses[-3] > 300 and mses[-4] > 300 and mses[-5] > 300 and mses[-6] > 300 and mses[-7] > 300 and mses[-8] > 300 and mses[-9] > 300 and mses[-10] > 300)):
                                
                                # if not privacy leaked
                                if not converged:
                                    converged = False # redundancy 
                                
                                plot_iters(is_last=True)
                                print("not converged!")

                                # tracking metrics
                                metrics[method][initializer][defense][magnitude]["failures"] +=1

                                # the FB dirty initializer always updates the "new" dummy_data
                                # and at first iteration must get it from "random" initializer
                                if exp == 0 and "FB-NF" in initializers:
                                    dummy_data[method]["FB-NF"][defense][magnitude]["new"] = dummy_data[method]["random"][defense][magnitude]["new"].clone().detach().requires_grad_(True)
                                
                                # fed back must skip if not converged, because it is clean
                                if initializer == "FB":
                                    dummy_data[method]["FB"][defense][magnitude]["failed"] = True

                                break
                        
                        # tracking metrics 
                        metrics[method][initializer][defense][magnitude]["losses_history"] = losses
                        metrics[method][initializer][defense][magnitude]["mses_history"] = mses
                        metrics[method][initializer][defense][magnitude]["iterations_history"] = train_iters[-1]
                        metrics[method][initializer][defense][magnitude]["iter_privacy_leaked"] = iter_privacy_leaked
                        metrics[method][initializer][defense][magnitude]["gt_label_history"] = gt_label[0].detach().cpu().tolist()
                        metrics[method][initializer][defense][magnitude]["pred_logit_history"] = pred_logit_history
                        if method == "DLG":
                            metrics[method][initializer][defense][magnitude]["dummy_logit_history"] = dummy_logit_history

                        # simplifying some metrics
                        if method == "DLG":
                            dummy_label_metric = torch.argmax(torch.from_numpy(np.array(metrics[method][initializer][defense][magnitude]["dummy_logit_history"][-1][-1]))).tolist()
                        elif method == "iDLG":
                            dummy_label_metric = metrics[method][initializer][defense][magnitude]["dummy_logit_history"][-1].tolist()
                        pred_label_metric = torch.argmax(torch.from_numpy(np.array(metrics[method][initializer][defense][magnitude]["pred_logit_history"][-1][-1]))).tolist()

                        # Saving to csv file
                        with open(f"{csv_save_path}/metrics_{dataset}_at_{INITIAL_TIME_STR}.csv", "a") as f:
                            write = csv.writer(f)
                            write.writerow([ imidx_list[-1], method, initializer, defense, magnitude, exp, iters, iter_privacy_leaked, gt_label[0].tolist(), dummy_label_metric, pred_label_metric, converged, current_loss, mses[-1]])
                            

                        def print_metrics():
                            # printing all final results
                            print("============metrics============")
                            print("imidx_list: ", imidx_list[-1])
                            print("method: ", method, ', initializer:', initializer, ', defense:', defense, ', magnitude:', magnitude)
                            print("exp: ", exp)
                            print("iters: ", train_iters[-1])
                            print("iter_privacy_leaked: ", metrics[method][initializer][defense][magnitude]["iter_privacy_leaked"])
                            print("loss: ", metrics[method][initializer][defense][magnitude]["losses_history"][-1])
                            print("mse: ", metrics[method][initializer][defense][magnitude]["mses_history"][-1])
                            print("gt_label: ", metrics[method][initializer][defense][magnitude]["gt_label_history"])
                            print("dummy_label: ", dummy_label_metric)
                            print("pred_label: ", pred_label_metric)
                            print("Total convergence: ", metrics[method][initializer][defense][magnitude]["convergences"])
                            print("Total failures: ", metrics[method][initializer][defense][magnitude]["failures"])
                            print("Initial time: ", INITIAL_TIME.strftime(date_print_format))
                            print("Final time: ", datetime.datetime.now().strftime(date_print_format))
                            print("===============================\n\n")

                        print_metrics()




if __name__ == '__main__':
    main()


