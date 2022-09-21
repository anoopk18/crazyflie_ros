import numpy as np
from torch.nn import functional as F
from solvers import *
from models import *
import sys
import glob
import os
import time
import tqdm

sys.path.insert(1, '../')
train_verbose_header = '\033[33m' + "[Trainer] " + '\033[0m'  # yellow color

def sample_and_grow(ode_train, traj_list, epochs, LR, lookahead, l2_lambda, plot_freq=50,
                    save_path=None, step_skip=1):
    state_dim = 6  # for rigid model the state is 6D
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ode_train.parameters()), lr=LR)
    for i in tqdm.tqdm(range(epochs)):
        for idx, true_traj in enumerate(traj_list):
            n_segments, _, n_state = true_traj.size()
            true_segments_list = []

            for j in range(0, n_segments - lookahead + 2 - BATCH_SKIP, BATCH_SKIP):
                j = j + i % BATCH_SKIP
                true_sampled_time_segment = (Tensor(np.arange(lookahead)) * step_skip).detach()
                true_sampled_segment = true_traj[j:j + lookahead]
                true_segments_list.append(true_sampled_segment)
            # concatenating all batches together
            obs = torch.cat(true_segments_list, 1)

            pred_traj = []
            ode_input = obs[0, :, :].unsqueeze(1)  # initial condition has size [1499, 1, 17]
            pred_traj.append(ode_input)
            for k in range(len(true_sampled_segment) - 1):
                z1 = ode_train(ode_input, Tensor(np.arange(2))).squeeze(1)
                ode_input = torch.cat([z1[:, :state_dim].unsqueeze(1), obs[k + 1, :, state_dim:].unsqueeze(1)], 2)
                pred_traj.append(ode_input)

            # prediction has size [plot_len, 1, 17]
            pred_traj = torch.swapaxes(torch.cat(pred_traj, 1), 0, 1)
            l2_norm = sum(p.pow(2.0).sum() for p in ode_train.parameters())
            # l2_lambda = 7e-8
            if idx == 0:
                # loss from the first trajectory
                loss = F.mse_loss(pred_traj[:, :, :state_dim], obs[:, :, :state_dim]) + l2_lambda* l2_norm
            else:
                # adding loss from other trajectories
                loss += F.mse_loss(pred_traj[:, :, :state_dim], obs[:, :, :state_dim]) + l2_lambda + l2_norm

        train_loss_arr.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        n_old_layers = ode_train.func.n_old_layers
        # freezing the weights of old layers
        for m in range(n_old_layers):
            if str(ode_train.func.nn_model[m])[:6] == 'Linear':
                ode_train.func.nn_model[m].weight.grad *= 0
                ode_train.func.nn_model[m].bias.grad *= 0
        
        optimizer.step()

        if i % plot_freq == 0:
            if i == 400:  # reduce learning rate after 500 epochs
                LR = 0.001
            elif i == 13000:
                LR = 0.01
            optimizer.param_groups[0]['lr'] = LR

            # computing trajectory using the current model
            plot_title = "\nIteration: {0} Step Size: {1} No. of Points: {2} Lookahead: {3} LR: {4}\n    Training " \
                         "Loss: {5:.3e}"
            print(plot_title.format(i, step_size, n_segments, LOOKAHEAD, LR, loss.item()))
            print("first and last param std:\n", torch.std(ode_train.func.nn_model[0].weight).detach().numpy(), torch.std(ode_train.func.nn_model[-1].weight).detach().numpy())
            if save_path is not None:
                torch.save({'ode_train': ode_train, 
                            'train_loss_arr': train_loss_arr}, save_path + "_epoch" + str(i) + ".pth")



def find_latest_data(data_path):
    """
    find the latest saved data and return its index
    """
    data_cnt = 0  # the first index to start looking
    while os.path.exists(data_path + "online_data" + str(data_cnt) + ".npy"):
        data_cnt += 1
    return data_cnt

    
if __name__ == '__main__':
    # KNODE parameters
    ode_solve = RK
    step_size = 1/400
    torch.manual_seed(0)

    model_cnt = 0
    ode_train = NeuralODE(RigidHybridAdditive(), ode_solve, step_size)
    
    training_data_path_list = ["OfflineData/training_data.npy"]
    train_traj_list = []
    # concatenating all trajectories into a list
    for i, data_path in enumerate(training_data_path_list):
        with open(data_path, 'rb') as f:
            train_set = np.load(f)
            print("Training traj {0} has shape: \n".format(i), train_set.shape)
        train_set = Tensor(train_set)
        print("train set shape is: ", train_set.shape)
        train_traj = train_set.detach().unsqueeze(1)
        train_traj = train_traj + torch.randn_like(train_traj) * 0.00
        train_traj_list.append(train_traj)

    # training parameters
    DEBUG = False
    step_skip = 1  # number of interpolations between observations
    train_loss_arr = []
    # save_path = 'SavedModels/ral_rr_phys_exp_smaller_circle.pth'  # path for saving the model
    save_path = 'OfflineModels/offline_model'  # path for saving the model
    ITER_OFFSET = 0
    BATCH_SKIP = 1
    EPOCHs = 1000  # No. of epochs to optimize
    LOOKAHEAD = 2  # alpha, the number of steps to lookahead
    name = "lookahead_" + str(LOOKAHEAD - 1)
    LR = 0.01  # optimization step size
    plot_freq = 50
    l2_lambda = 0e-7
    sample_and_grow(ode_train, train_traj_list, EPOCHs, LR, LOOKAHEAD, l2_lambda,
                    plot_freq=plot_freq, save_path=save_path, step_skip=step_skip)
