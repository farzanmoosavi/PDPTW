# import datetime
# import numpy as np
# import torch
# import os
# import time
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from VRP_Actor import Model
# from creat_vrp import create_data, reward1
# from collections import OrderedDict
# from collections import namedtuple
# from itertools import product
# from torch.optim.lr_scheduler import LambdaLR
# from rolloutBaseline1 import RolloutBaseline
# from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
# import torch.utils.checkpoint as checkpoint
#
# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:54"
# # torch.cuda.empty_cache()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cpu')
# n_nodes = 50
# depots = 2  # included in n_nodes
# n_robots = 2
# n_drones = 2
# print(device)
# steps = n_nodes + depots
#
# def clip_grad_norms(param_groups, max_norm=math.inf):
#
#     grad_norms = [
#         torch.nn.utils.clip_grad_norm_(
#             group['params'],
#             max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
#             norm_type=2
#         )
#         for group in param_groups
#     ]
#     grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
#     return grad_norms, grad_norms_clipped
# def rollout(model, dataset, num_drones, num_robots, steps):
#     model.eval()
#
#     def eval_model_bat(bat):
#         with torch.no_grad():
#             cost, _, Time, BL = model(bat, num_drones, num_robots, True)
#
#             cost = reward1(bat.time_window, cost.detach(), bat.edge_attr_d.detach(), bat.edge_attr_r.detach(), Time.detach(), BL.detach(), num_drones)
#         return cost.cpu()
#
#     totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
#     return totall_cost
#
#
# max_grad_norm = 2
#
# rewardss = []
#
#
# def adv_normalize(adv):
#     std = adv.std()
#     assert std != 0. and not torch.isnan(std), 'Need nonzero std'
#     n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
#     return n_advs
#
#
# def train():
#     # ------------------------------------------------------------------------------------------------------------------------------
#     class RunBuilder():
#
#         @staticmethod
#         def get_runs(params):
#             Run = namedtuple('Run', params.keys())
#             runs = []
#             for v in product(*params.values()):
#                 runs.append(Run(*v))
#             return runs
#
#
#     params = OrderedDict(
#         lr=[1e-3],
#         batch_size=[512],
#         hidden_node_dim=[128],
#         hidden_edge_dim=[16],
#         conv_laysers=[4],
#         data_size=[1280000]
#
#     )
#     runs = RunBuilder.get_runs(params)
#     # -------------------------------------------------------------------------------------------------------------------------------------
#
#
#     folder = 'Vrp-{}-GAT'.format(n_nodes)
#     filename = 'rollout'
#     for lr, batch_size, hidden_node_dim, hidden_edge_dim, conv_laysers, data_size in runs:
#         print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'conv_laysers:', lr, batch_size,
#               hidden_node_dim, hidden_edge_dim, conv_laysers)
#         data_loder = create_data(n_nodes, depots, n_robots, n_drones, data_size, batch_size)
#         valid_loder = create_data(n_nodes, depots, n_robots, n_drones, 10000, batch_size)
#         print('Data creation completed')
#
#         actor = Model(4, hidden_node_dim, 1, hidden_edge_dim, conv_laysers=conv_laysers).to(device)
#         rol_baseline = RolloutBaseline(actor, valid_loder, n_robots=n_robots, n_drones=n_drones, n_nodes=steps)
#         # initWeights(actor)
#         filepath = os.path.join(folder, filename)
#         '''path = os.path.join(filepath,'%s' % 3)
#                 if os.path.exists(path):
#                     path1 = os.path.join(path, 'actor.pt')
#                     self.agent.old_polic.load_state_dict(torch.load(path1, device))'''
#         actor_optim = optim.Adam(actor.parameters(), lr=lr)
#         scaler = GradScaler()  # Mixed precision scaler
#
#         costs = []
#         for epoch in range(100):
#             print("epoch:", epoch, "------------------------------------------------")
#             actor.train()
#
#             times, losses, rewards, critic_rewards = [], [], [], []
#             epoch_start = time.time()
#             start = epoch_start
#
#             scheduler = LambdaLR(actor_optim, lr_lambda=lambda f: 0.96 ** epoch)
#             actor_optim.zero_grad()
#
#             for batch_idx, batch in enumerate(data_loder):
#                 batch = batch.to(device)
#                 with autocast():
#                     tour_indices, tour_logp, Time, BL = actor(batch, n_drones, n_robots)
#
#                     rewar = reward1(batch.time_window, tour_indices.detach(), batch.edge_attr_d.detach(), batch.edge_attr_r.detach(), Time.detach(),
#                                     BL.detach(), n_drones)
#                     base_reward = rol_baseline.eval(batch, n_drones, n_robots, steps)
#
#                     advantage = (rewar - base_reward)
#                     if not advantage.ne(0).any():
#                         print("advantage==0.")
#                     advantage = adv_normalize(advantage)
#                     actor_loss = torch.mean(advantage.detach() * tour_logp)
#                 # print(actor_loss,actor_loss.size(
#
#                 # ))
#                 # actor_optim.zero_grad()
#                 # actor_loss.backward()
#                 # grad_norms = clip_grad_norms(actor_optim.param_groups, 1)
#                 torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
#                 # actor_optim.step()
#                 # scheduler.step()
#                 # rewards.append(torch.mean(rewar.detach()).item())
#                 # losses.append(torch.mean(actor_loss.detach()).item())
#                 scheduler.step()
#                 scaler.scale(actor_loss).backward()  # Mixed precision scaling
#                 scaler.step(actor_optim)
#                 scaler.update()
#
#
#
#                 rewards.append(torch.mean(rewar.detach()).item())
#                 losses.append(torch.mean(actor_loss.detach()).item())
#
#                 # Explicitly delete variables to free memory
#                 del batch, tour_indices, tour_logp, Time, BL, rewar, base_reward, advantage, actor_loss
#                 torch.cuda.empty_cache()
#
#                 step = 10
#                 if (batch_idx + 1) % step == 0:
#                     end = time.time()
#                     times.append(end - start)
#                     start = end
#
#                     mean_loss = np.mean(losses[-step:])
#                     mean_reward = np.mean(rewards[-step:])
#
#                     print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
#                           (batch_idx, len(data_loder), mean_reward, mean_loss,
#                            times[-1]))
#
#             rol_baseline.epoch_callback(actor, epoch)
#
#             epoch_dir = os.path.join(filepath, '%s' % epoch)
#             if not os.path.exists(epoch_dir):
#                 os.makedirs(epoch_dir)
#             save_path = os.path.join(epoch_dir, 'actor.pt')
#             torch.save(actor.state_dict(), save_path)
#             cost = rollout(actor, valid_loder, n_drones, n_robots, steps)
#             cost = cost.mean()
#             costs.append(cost.item())
#             np.savetxt('myarray.txt', costs)
#
#             print('Problem:PDPTW''%s' % n_nodes, '/ Average Cost:', cost.item())
#             print(costs)
#
#
# train()


import datetime
import numpy as np
import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from VRP_Actor import Model
import math
from creat_vrp import create_data, reward1
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from torch.optim.lr_scheduler import LambdaLR
from rolloutBaseline1 import RolloutBaseline
from torch.amp import autocast

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:54"
# torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
n_nodes = 100
depots = 3  # included in n_nodes
n_robots = 3
n_drones = 2
print(device)
steps = n_nodes + depots


def measure_time(start_time, name):
    end_time = time.time()
    print(f"{name} took: {end_time - start_time:.4f} seconds")
    return end_time


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def rollout(model, dataset, num_drones, num_robots, steps):
    model.eval()

    def eval_model_bat(bat):
        #     with torch.no_grad():
        #         cost, _, Time,BL = model(bat,num_drones,num_robots,True)

        #         cost = reward1(bat.time_window, cost.detach(),bat.edge_attr_d,bat.edge_attr_r,Time,BL,num_drones)
        #     return cost.cpu()
        # totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
        # return totall_cost
        bat = {key: value.to(device) for key, value in bat.items() if isinstance(value, torch.Tensor)}
        with torch.no_grad():
            cost, _, Time, BL = model(bat, num_drones, num_robots, True)
            cost = reward1(bat['time_window'], cost.detach(), bat['edge_attr_d'], bat['edge_attr_r'], Time.detach(),
                           BL.detach(), num_drones)
        return cost.cpu()

    # Apply the evaluation to each batch in the dataset
    total_cost = torch.cat([eval_model_bat(bat) for bat in dataset], 0)
    return total_cost


max_grad_norm = 2

rewardss = []


def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs


def train():
    # ------------------------------------------------------------------------------------------------------------------------------
    class RunBuilder():

        @staticmethod
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    params = OrderedDict(
        lr=[1e-4],
        batch_size=[128],
        hidden_node_dim=[128],
        hidden_edge_dim=[16],
        conv_laysers=[4],
        data_size=[5120]

    )
    runs = RunBuilder.get_runs(params)
    # -------------------------------------------------------------------------------------------------------------------------------------

    folder = 'Vrp-{}-GAT'.format(n_nodes)
    filename = 'rollout'
    for lr, batch_size, hidden_node_dim, hidden_edge_dim, conv_laysers, data_size in runs:
        print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'conv_laysers:', lr, batch_size,
              hidden_node_dim, hidden_edge_dim, conv_laysers)
        data_loder = create_data(n_nodes, depots, n_robots, n_drones, data_size, batch_size=batch_size, num_workers=0,
                                 use_cache=True, cache_file="dataset_cache.pkl")
        valid_loder = create_data(n_nodes, depots, n_robots, n_drones, 512, batch_size=batch_size, num_workers=0,
                                  use_cache=True, cache_file="dataset_cache1.pkl")
        print('Data creation completed')

        actor = Model(4, hidden_node_dim, 1, hidden_edge_dim, conv_laysers=conv_laysers).to(device)
        rol_baseline = RolloutBaseline(actor, valid_loder, n_robots=n_robots, n_drones=n_drones, n_nodes=steps)
        # initWeights(actor)
        filepath = os.path.join(folder, filename)
        '''path = os.path.join(filepath,'%s' % 3)
                if os.path.exists(path):
                    path1 = os.path.join(path, 'actor.pt')
                    self.agent.old_polic.load_state_dict(torch.load(path1, device))'''

        actor_optim = optim.Adam(actor.parameters(), lr=lr)

        # scaler = GradScaler()  # Mixed precision scaler
        costs = []
        for epoch in range(100):
            print("epoch:", epoch, "------------------------------------------------")
            actor.train()
            scheduler = LambdaLR(actor_optim, lr_lambda=lambda f: 0.96 ** epoch)
            times, losses, rewards, critic_rewards = [], [], [], []
            epoch_start = time.time()
            start = epoch_start

            for batch_idx, batch in enumerate(data_loder):

                batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}

                # with autocast(device_type='cuda'):
                forward_start = time.time()
                tour_indices, tour_logp, Time, BL = actor(batch, n_drones, n_robots, greedy=False, T=1,
                                                          checkpoint_encoder=True, training=True)
                measure_time(forward_start, "Forward pass")
                reward_start = time.time()
                rewar = reward1(batch['time_window'], tour_indices.detach(), batch['edge_attr_d'],
                                batch['edge_attr_r'], Time.detach(),
                                BL.detach(), n_drones)
                base_reward = rol_baseline.eval(batch, n_drones, n_robots, steps)
                measure_time(reward_start, "Reward and baseline evaluation")

                advantage = (rewar - base_reward)
                if not advantage.ne(0).any():
                    print("advantage==0.")
                advantage = adv_normalize(advantage)
                actor_loss = torch.mean(advantage.detach() * tour_logp)

                # print(actor_loss,actor_loss.size(

                # actor_optim.zero_grad()
                # ))
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()
                scheduler.step()
                # rewards.append(torch.mean(rewar.detach()).item())
                # losses.append(torch.mean(actor_loss.detach()).item())

                # scaler.scale(actor_loss).backward()  # Mixed precision scaling
                # grad_norms = clip_grad_norms(actor_optim.param_groups, 1)
                # scaler.step(actor_optim)
                # scaler.update()
                # scheduler.step()

                rewards.append(torch.mean(rewar.detach()).item())
                losses.append(torch.mean(actor_loss.detach()).item())

                # Explicitly delete variables to free memory
                del tour_indices, tour_logp, Time, BL, rewar, base_reward, advantage, actor_loss
                # torch.cuda.empty_cache()

                step = 20
                if (batch_idx + 1) % step == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end

                    mean_loss = np.mean(losses[-step:])
                    mean_reward = np.mean(rewards[-step:])

                    print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                          (batch_idx, len(data_loder), mean_reward, mean_loss,
                           times[-1]))
            rol_baseline.epoch_callback(actor, epoch)

            epoch_dir = os.path.join(filepath, '%s' % epoch)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)
            cost = rollout(actor, valid_loder, n_drones, n_robots, steps)
            cost = cost.mean()
            costs.append(cost.item())
            np.savetxt('myarray.txt', costs)

            print('Problem:PDPTW''%s' % n_nodes, '/ Average Cost:', cost.item())
            print(costs)


train()
