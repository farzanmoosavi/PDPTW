import torch
from creat_vrp import reward1
from scipy.stats import ttest_rel
import copy
import numpy as np

from torch.nn import DataParallel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def rollout1(model, dataset, num_robots, num_drones, n_nodes):
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, time, charge = model(bat, num_drones, num_robots, True)
            cost = reward1(bat.time_window, cost.detach(), bat.edge_attr_d.detach(), bat.edge_attr_r.detach(), time.detach(), charge.detach(), num_drones)
        return cost.cpu()

    totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return totall_cost


class RolloutBaseline():

    def __init__(self, model, dataset, n_robots, n_drones, n_nodes=50, epoch=0):
        super(RolloutBaseline, self).__init__()
        self.n_nodes = n_nodes
        self.n_robots = n_robots
        self.n_drones = n_drones
        self.dataset = dataset
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        self.bl_vals = rollout1(self.model, self.dataset, self.n_robots, self.n_drones,
                                n_nodes=self.n_nodes).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, x, num_drones, num_robots, n_nodes):

        with torch.no_grad():
            tour, _, time, charge = self.model(x, num_drones, num_robots, True)
            v = reward1(x.time_window, tour.detach(), x.edge_attr_d.detach(), x.edge_attr_r.detach(), time.detach(), charge.detach(), num_drones)

        # There is no loss
        return v

    def epoch_callback(self, model, epoch):

        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout1(model, self.dataset, self.n_robots, self.n_drones, self.n_nodes).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals, axis=0)
            p_val = p / 2  # one-sided

            assert np.any(t < 0), "T-statistic should be negative"
            print("p-values: {}".format(p_val))

            if np.all(p_val < 0.05):
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])
