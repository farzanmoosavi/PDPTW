# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# import math
# import numpy as np
#
# def PC_d(m_pl, v_g):
#
#     nu = 0.9
#     rho = 1.225
#     M = 12
#     C_d1 = 1.49
#     C_d2 = 2.2
#     A1 = 0.224
#     A2 = 0.1
#     zeta = 1.4
#     n = 8
#     v_w = 15*np.random.random() * torch.rand_like(v_g)
#     psi_w = math.pi*2*np.random.random()
#     psi = 0
#
#     x = v_g * torch.cos(torch.tensor(psi)) + v_w * torch.cos(torch.tensor(psi_w))
#     z = v_g * torch.sin(torch.tensor(psi)) + v_w * torch.sin(torch.tensor(psi_w))
#     chi = torch.atan(z / x)
#     v_a = torch.sqrt(2 * v_g**2 + 2 * v_w**2 - 2 * v_w * v_g * torch.cos(torch.tensor(psi_w) - chi))
#
#     Drag = rho * (C_d1 * A1 + C_d2 * A2) * v_a**2 / 2
#     W = 9.8 * (M + m_pl)
#     T = Drag + W
#     alpha = torch.atan(Drag / W)
#
#     def equation(vi, W, v_a, alpha):
#         return vi - W / (2 * n * rho * zeta * torch.sqrt((v_a * torch.cos(alpha))**2 + (v_a * torch.sin(alpha) + vi)**2))
#
#     def newton_raphson(vi_initial, W, v_a, alpha, tol=1e-6, max_iter=100):
#         vi = vi_initial
#         for _ in range(max_iter):
#             f = equation(vi, W, v_a, alpha)
#             f_prime = 1 + W / (2 * n * rho * zeta * ((v_a * torch.cos(alpha))**2 + (v_a * torch.sin(alpha) + vi) / torch.sqrt((v_a * torch.cos(alpha))**2 + (v_a * torch.sin(alpha) + vi)**2)))
#             vi_next = vi - f / f_prime
#             if torch.abs(vi_next - vi).max() < tol:
#                 break
#             vi = vi_next
#         return vi
#     if v_g.numel() == 0:
#       v_i = 0
#     else:
#       vi_initial_guess = torch.full_like(v_g, 10.0)
#       v_i = newton_raphson(vi_initial_guess, W, v_a, alpha)
#
#     P = T * (v_a * torch.sin(alpha) + v_i) / nu / 1000
#
#     return P
#
# def PC_r(m_pl,v_g):
#
#     C_r = 0.25
#     nu = 0.8
#     M = 30
#     P = C_r*(M+m_pl)*9.8*v_g/nu/1000
#
#     return P
#
#
# def update_state(demands, time_window, battery, T_t, capacity, selected, E, C, num_drones, actions, edge_attr_r,
#                  edge_attr_d, i):
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch_size, n_agents = selected.size()
#     num_depots = torch.sum(demands == 0).item() // batch_size
#     num_nodes = demands.size(1)
#     # num_pickup = (num_nodes - num_depots) // 2
#
#     batch_d = torch.arange(batch_size).unsqueeze(1).expand(-1, num_drones)
#     batch_r = torch.arange(batch_size).unsqueeze(1).expand(-1, n_agents - num_drones)
#
#     depot = selected.squeeze(-1).lt(num_depots)  # Is there a group to access the depot
#     previous_indices = actions[-2].squeeze(2)
#     current_indices = actions[-1].squeeze(2)
#
#     current_demand = torch.gather(demands, 1, selected)
#     current_time_window = torch.gather(time_window, 1, current_indices)
#     previous_time_window = torch.gather(time_window, 1, previous_indices)
#
#     delta_t = (current_time_window - T_t.squeeze(2))
#
#     dynamic_capacity = capacity.squeeze(2) - current_demand
#
#     dis_d = edge_attr_d[batch_d, previous_indices[:, :num_drones], current_indices[:, :num_drones]]
#     dis_r = edge_attr_r[batch_r, previous_indices[:, num_drones:], current_indices[:, num_drones:]]
#
#     zero_delta_t_d = delta_t[:, 0:num_drones] == 0
#     zero_delta_t_r = delta_t[:, num_drones:] == 0
#
#     vel_d = torch.zeros_like(dis_d)
#     vel_r = torch.zeros_like(dis_r)
#     time_d = torch.zeros_like(dis_d)
#     time_r = torch.zeros_like(dis_r)
#     battery_d = torch.zeros_like(dis_d)
#     battery_r = torch.zeros_like(dis_r)
#
#     vel_d[~zero_delta_t_d] = torch.div(dis_d[~zero_delta_t_d], torch.abs(delta_t[:, 0:num_drones][~zero_delta_t_d]))
#     vel_r[~zero_delta_t_r] = torch.div(dis_r[~zero_delta_t_r], torch.abs(delta_t[:, num_drones:][~zero_delta_t_r]))
#
#     vel_d[zero_delta_t_d] = 0.8
#     vel_r[zero_delta_t_r] = 0.4
#
#     v_dmax = 1.2
#     v_rmax = 0.5
#
#     vel_d = torch.clamp(vel_d, max=v_dmax)
#     vel_r = torch.clamp(vel_r, max=v_rmax)
#
#     positive_delta_t_d = delta_t[:, 0:num_drones].gt(0)
#     positive_delta_t_r = delta_t[:, num_drones:].gt(0)
#     negative_delta_t_d = delta_t[:, 0:num_drones].le(0)
#     negative_delta_t_r = delta_t[:, num_drones:].le(0)
#
#     if positive_delta_t_d.any():
#         time_d[positive_delta_t_d] = T_t.squeeze(2)[:, 0:num_drones][positive_delta_t_d] + torch.abs(
#             delta_t[:, 0:num_drones][positive_delta_t_d])
#         time_d[vel_d.ge(v_dmax) & positive_delta_t_d] = T_t.squeeze(2)[:, 0:num_drones][
#                                                             vel_d.ge(v_dmax) & positive_delta_t_d] + dis_d[
#                                                             vel_d.ge(v_dmax) & positive_delta_t_d] / v_dmax
#         battery_d[positive_delta_t_d] = battery.squeeze(2)[:, 0:num_drones][positive_delta_t_d] - (
#                 delta_t[:, 0:num_drones][positive_delta_t_d] * PC_d(
#             current_demand[:, 0:num_drones][positive_delta_t_d], vel_d[positive_delta_t_d] * 50 / 3)) * 60
#         battery_d[vel_d.ge(v_dmax) & positive_delta_t_d] = battery.squeeze(2)[:, 0:num_drones][
#                                                                vel_d.ge(v_dmax) & positive_delta_t_d] - (dis_d[vel_d.ge(
#             v_dmax) & positive_delta_t_d] / v_dmax * PC_d(
#             current_demand[:, 0:num_drones][vel_d.ge(v_dmax) & positive_delta_t_d],
#             v_dmax * torch.ones_like(vel_d)[vel_d.ge(v_dmax) & positive_delta_t_d] * 50 / 3)) * 60
#
#     if positive_delta_t_r.any():
#         time_r[positive_delta_t_r] = T_t.squeeze(2)[:, num_drones:][positive_delta_t_r] + torch.abs(
#             delta_t[:, num_drones:][positive_delta_t_r])
#         time_r[vel_r.ge(v_rmax) & positive_delta_t_r] = T_t.squeeze(2)[:, num_drones:][
#                                                             vel_r.ge(v_rmax) & positive_delta_t_r] + dis_r[
#                                                             vel_r.ge(v_rmax) & positive_delta_t_r] / v_rmax
#         battery_r[positive_delta_t_r] = battery.squeeze(2)[:, num_drones:][positive_delta_t_r] - (
#                 delta_t[:, num_drones:][positive_delta_t_r] * PC_r(
#             current_demand[:, num_drones:][positive_delta_t_r], vel_r[positive_delta_t_r] * 50 / 3)) * 60
#         battery_r[vel_r.ge(v_rmax) & positive_delta_t_r] = battery.squeeze(2)[:, num_drones:][
#                                                                vel_r.ge(v_rmax) & positive_delta_t_r] - (dis_r[vel_r.ge(
#             v_rmax) & positive_delta_t_r] / v_rmax * PC_r(
#             current_demand[:, num_drones:][vel_r.ge(v_rmax) & positive_delta_t_r],
#             v_rmax * torch.ones_like(vel_r)[vel_r.ge(v_rmax) & positive_delta_t_r] * 50 / 3)) * 60
#
#     if negative_delta_t_d.any():
#         time_d[negative_delta_t_d] = T_t.squeeze(2)[:, 0:num_drones][negative_delta_t_d] + dis_d[
#             negative_delta_t_d] / v_dmax
#         battery_d[negative_delta_t_d] = battery.squeeze(2)[:, 0:num_drones][negative_delta_t_d] - (
#                 dis_d[negative_delta_t_d] / v_dmax * PC_d(current_demand[:, 0:num_drones][negative_delta_t_d],
#                                                           v_dmax * torch.ones_like(vel_d)[
#                                                               negative_delta_t_d] * 50 / 3)) * 60
#
#     if negative_delta_t_r.any():
#         time_r[negative_delta_t_r] = T_t.squeeze(2)[:, num_drones:][negative_delta_t_r] + dis_r[
#             negative_delta_t_r] / v_rmax
#         battery_r[negative_delta_t_r] = battery.squeeze(2)[:, num_drones:][negative_delta_t_r] - (
#                 dis_r[negative_delta_t_r] / v_rmax * PC_r(current_demand[:, num_drones:][negative_delta_t_r],
#                                                           v_rmax * torch.ones_like(vel_r)[
#                                                               negative_delta_t_r] * 50 / 3)) * 60
#
#     dynamic_battery = torch.cat((battery_d, battery_r), -1)
#     dynamic_time = torch.cat((time_d, time_r), -1)
#
#     if depot.any():
#         dynamic_battery[:, :num_drones][depot[:, :num_drones]] = E[0]
#         dynamic_battery[:, num_drones:][depot[:, num_drones:]] = E[1]
#         vel_d = 0.8
#         vel_r = 0.3
#         dynamic_time[:, :num_drones][depot[:, :num_drones]] = dis_d[depot[:, :num_drones]] / vel_d + \
#                                                               T_t.squeeze(2)[:, :num_drones][
#                                                                   depot[:, :num_drones]]
#         dynamic_time[:, num_drones:][depot[:, num_drones:]] = dis_r[depot[:, num_drones:]] / vel_r + \
#                                                               T_t.squeeze(2)[:, num_drones:][
#                                                                   depot[:, num_drones:]]
#     # + torch.randint(35, 45,
#     #                 (1,)).to(
#     #     device)
#     return dynamic_capacity.detach(), dynamic_time.detach(), dynamic_battery.detach()
#
#
# def update_mask(demand, time_window, capacity, T_t, battery, num_drones, selected, mask, E, i):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch_size, n_agents = selected.size()
#     num_depots = torch.sum(demand == 0).item() // batch_size
#     num_nodes = demand.size(1)
#     num_pickup = (num_nodes - num_depots) // 2
#
#     depot_indices = torch.arange(num_depots)
#     pickup_indices = torch.arange(num_depots, num_depots + num_pickup)
#     delivery_indices = torch.arange(num_depots + num_pickup, num_nodes)
#
#     go_depot = selected.lt(num_depots)
#     pickup_start = num_depots
#     pickup_end = num_depots + num_pickup
#     go_pickup_agent = selected.ge(pickup_start) & selected.lt(pickup_end)
#     go_delivery_agent = selected.ge(num_depots + num_pickup) & selected.lt(num_nodes)
#
#     mask_ = mask.clone()
#     mask = mask_.scatter_(2, selected.unsqueeze(-1), 1)
#     maskfil = (mask.max(dim=1)[0])
#     mask2 = mask.clone()
#     maskfil = maskfil.unsqueeze(1).expand(batch_size, n_agents, num_nodes)
#
#     if (~go_depot).any():
#         for depot in range(num_depots):
#             mask2[(~go_depot).nonzero(as_tuple=True)[0], (~go_depot).nonzero(as_tuple=True)[1], depot] = 1
#
#     corresponding_deliveries = num_depots + num_pickup + (pickup_indices - num_depots)
#
#     mask[:, :, depot_indices] = torch.where(go_depot.unsqueeze(-1), 1, mask[:, :, depot_indices])
#
#     # Adjusted operation for correct broadcasting
#     unvisited_pickups = (mask2[:, :, pickup_indices] == 0)
#     mask[:, :, pickup_indices] *= ~unvisited_pickups
#
#     mask[:, :, delivery_indices] = 1
#     for p_idx, d_idx in zip(pickup_indices, corresponding_deliveries):
#         mask[:, :, d_idx] = torch.where(mask2[:, :, p_idx] == 1, 0, mask[:, :, d_idx])
#
#     mask = torch.where(maskfil == 1, maskfil, mask)
#
#     if i + 1 > demand.size(1):
#         is_done = (mask2[:, :, num_depots:].sum(2) >= (demand.size(1) - num_depots)).float()
#         combined = is_done.gt(num_depots)
#         for depot in range(num_depots):
#             mask2[combined.nonzero(as_tuple=True)[0], combined.nonzero(as_tuple=True)[1], depot] = 0
#
#     threshold_values = torch.cat([torch.ones_like(battery[:, 0:num_drones]) * E[0] * 0.40,
#                                   torch.ones_like(battery[:, num_drones:]) * E[1] * 0.25], 1)
#
#     time_window_exceeds_battery = time_window.unsqueeze(1) > (torch.cat(
#         [battery.squeeze(2)[:, 0:num_drones] / (60 * 1.5), battery.squeeze(2)[:, num_drones:] / (60 * 0.6)],
#         -1).unsqueeze(2) + T_t)
#
#     demand_exceeds_capacity = demand.unsqueeze(1) > capacity
#     final_mask = demand_exceeds_capacity + mask2 + time_window_exceeds_battery + mask
#
#     battery_exceeds_threshold = battery < threshold_values
#     if battery_exceeds_threshold.any():
#         for agent in range(n_agents):
#             final_mask[battery_exceeds_threshold[:, agent].nonzero(as_tuple=True)[0], agent, :num_depots] = 0
#             final_mask[battery_exceeds_threshold[:, agent].nonzero(as_tuple=True)[0], agent, num_depots:num_nodes] = 1
#
#     return final_mask.detach(), mask2.detach()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math
import numpy as np


def PC_d(m_pl, v_g):
    nu = 0.9
    rho = 1.225
    M = 12
    C_d1 = 1.49
    C_d2 = 2.2
    A1 = 0.224
    A2 = 0.1
    zeta = 1.4
    n = 8
    v_w = 20 * np.random.random() * torch.rand_like(v_g)
    psi_w = math.pi * 2 * np.random.random()
    psi = 0

    x = v_g * torch.cos(torch.tensor(psi)) + v_w * torch.cos(torch.tensor(psi_w))
    z = v_g * torch.sin(torch.tensor(psi)) + v_w * torch.sin(torch.tensor(psi_w))
    chi = torch.atan(z / x)
    v_a = torch.sqrt(2 * v_g ** 2 + 2 * v_w ** 2 - 2 * v_w * v_g * torch.cos(torch.tensor(psi_w) - chi))

    Drag = rho * (C_d1 * A1 + C_d2 * A2) * v_a ** 2 / 2
    W = 9.8 * (M + m_pl)
    T = Drag + W
    alpha = torch.atan(Drag / W)

    # def equation(vi, W, v_a, alpha):
    #     return vi - W / (2 * n * rho * zeta * torch.sqrt((v_a * torch.cos(alpha))**2 + (v_a * torch.sin(alpha) + vi)**2))
    #
    # def newton_raphson(vi_initial, W, v_a, alpha, tol=1e-6, max_iter=100):
    #     vi = vi_initial
    #     for _ in range(max_iter):
    #         f = equation(vi, W, v_a, alpha)
    #         f_prime = 1 + W / (2 * n * rho * zeta * ((v_a * torch.cos(alpha))**2 + (v_a * torch.sin(alpha) + vi) / torch.sqrt((v_a * torch.cos(alpha))**2 + (v_a * torch.sin(alpha) + vi)**2)))
    #         vi_next = vi - f / f_prime
    #         if torch.abs(vi_next - vi).max() < tol:
    #             break
    #         vi = vi_next
    #     return vi
    # if v_g.numel() == 0:
    #   v_i = 0
    # else:
    #   vi_initial_guess = torch.full_like(v_g, 10.0)
    #   v_i = newton_raphson(vi_initial_guess, W, v_a, alpha)

    P = T * (v_a * torch.sin(alpha) + 1) / nu / 1000

    return P


def PC_r(m_pl, v_g):
    C_r = 0.25
    nu = 0.8
    M = 30
    P = C_r * (M + m_pl) * 9.8 * v_g / nu / 1000

    return P


def update_state(demands, time_window, battery, T_t, capacity, selected, E, C, num_drones, actions, edge_attr_r,
                 edge_attr_d, i):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, n_agents = selected.size()
    num_depots = torch.sum(demands == 0).item() // batch_size
    # num_nodes = demands.size(1)
    # num_pickup = (num_nodes - num_depots) // 2

    batch_d = torch.arange(batch_size).unsqueeze(1).expand(-1, num_drones)
    batch_r = torch.arange(batch_size).unsqueeze(1).expand(-1, n_agents - num_drones)

    depot = selected.squeeze(-1).lt(num_depots)  # Is there a group to access the depot
    previous_indices = actions[-2].squeeze(2)
    current_indices = actions[-1].squeeze(2)

    current_demand = torch.gather(demands, 1, selected)
    current_time_window = torch.gather(time_window, 1, current_indices)
    # previous_time_window = torch.gather(time_window, 1, previous_indices)

    delta_t = (current_time_window - T_t.squeeze(2))

    dynamic_capacity = capacity.squeeze(2) - current_demand

    dis_d = edge_attr_d[batch_d, previous_indices[:, :num_drones], current_indices[:, :num_drones]]
    dis_r = edge_attr_r[batch_r, previous_indices[:, num_drones:], current_indices[:, num_drones:]]

    zero_delta_t_d = delta_t[:, 0:num_drones] == 0
    zero_delta_t_r = delta_t[:, num_drones:] == 0

    vel_d = torch.zeros_like(dis_d)
    vel_r = torch.zeros_like(dis_r)
    time_d = torch.zeros_like(dis_d)
    time_r = torch.zeros_like(dis_r)
    battery_d = torch.zeros_like(dis_d)
    battery_r = torch.zeros_like(dis_r)

    vel_d[~zero_delta_t_d] = torch.div(dis_d[~zero_delta_t_d], torch.abs(delta_t[:, 0:num_drones][~zero_delta_t_d]))
    vel_r[~zero_delta_t_r] = torch.div(dis_r[~zero_delta_t_r], torch.abs(delta_t[:, num_drones:][~zero_delta_t_r]))

    vel_d[zero_delta_t_d] = 0.8
    vel_r[zero_delta_t_r] = 0.4

    v_dmax = 1.2
    v_rmax = 0.5

    vel_d = torch.clamp(vel_d, max=v_dmax)
    vel_r = torch.clamp(vel_r, max=v_rmax)

    positive_delta_t_d = delta_t[:, 0:num_drones].gt(0)
    positive_delta_t_r = delta_t[:, num_drones:].gt(0)
    negative_delta_t_d = delta_t[:, 0:num_drones].le(0)
    negative_delta_t_r = delta_t[:, num_drones:].le(0)

    if positive_delta_t_d.any():
        time_d[positive_delta_t_d] = T_t.squeeze(2)[:, 0:num_drones][positive_delta_t_d] + torch.abs(
            delta_t[:, 0:num_drones][positive_delta_t_d])
        time_d[vel_d.ge(v_dmax) & positive_delta_t_d] = T_t.squeeze(2)[:, 0:num_drones][
                                                            vel_d.ge(v_dmax) & positive_delta_t_d] + dis_d[
                                                            vel_d.ge(v_dmax) & positive_delta_t_d] / v_dmax
        battery_d[positive_delta_t_d] = battery.squeeze(2)[:, 0:num_drones][positive_delta_t_d] - (
                delta_t[:, 0:num_drones][positive_delta_t_d] * PC_d(
            current_demand[:, 0:num_drones][positive_delta_t_d], vel_d[positive_delta_t_d] * 50 / 3)) * 60
        battery_d[vel_d.ge(v_dmax) & positive_delta_t_d] = battery.squeeze(2)[:, 0:num_drones][
                                                               vel_d.ge(v_dmax) & positive_delta_t_d] - (dis_d[vel_d.ge(
            v_dmax) & positive_delta_t_d] / v_dmax * PC_d(
            current_demand[:, 0:num_drones][vel_d.ge(v_dmax) & positive_delta_t_d],
            v_dmax * torch.ones_like(vel_d)[vel_d.ge(v_dmax) & positive_delta_t_d] * 50 / 3)) * 60

    if positive_delta_t_r.any():
        time_r[positive_delta_t_r] = T_t.squeeze(2)[:, num_drones:][positive_delta_t_r] + torch.abs(
            delta_t[:, num_drones:][positive_delta_t_r])
        time_r[vel_r.ge(v_rmax) & positive_delta_t_r] = T_t.squeeze(2)[:, num_drones:][
                                                            vel_r.ge(v_rmax) & positive_delta_t_r] + dis_r[
                                                            vel_r.ge(v_rmax) & positive_delta_t_r] / v_rmax
        battery_r[positive_delta_t_r] = battery.squeeze(2)[:, num_drones:][positive_delta_t_r] - (
                delta_t[:, num_drones:][positive_delta_t_r] * PC_r(
            current_demand[:, num_drones:][positive_delta_t_r], vel_r[positive_delta_t_r] * 50 / 3)) * 60
        battery_r[vel_r.ge(v_rmax) & positive_delta_t_r] = battery.squeeze(2)[:, num_drones:][
                                                               vel_r.ge(v_rmax) & positive_delta_t_r] - (dis_r[vel_r.ge(
            v_rmax) & positive_delta_t_r] / v_rmax * PC_r(
            current_demand[:, num_drones:][vel_r.ge(v_rmax) & positive_delta_t_r],
            v_rmax * torch.ones_like(vel_r)[vel_r.ge(v_rmax) & positive_delta_t_r] * 50 / 3)) * 60

    if negative_delta_t_d.any():
        time_d[negative_delta_t_d] = T_t.squeeze(2)[:, 0:num_drones][negative_delta_t_d] + dis_d[
            negative_delta_t_d] / v_dmax
        battery_d[negative_delta_t_d] = battery.squeeze(2)[:, 0:num_drones][negative_delta_t_d] - (
                dis_d[negative_delta_t_d] / v_dmax * PC_d(current_demand[:, 0:num_drones][negative_delta_t_d],
                                                          v_dmax * torch.ones_like(vel_d)[
                                                              negative_delta_t_d] * 50 / 3)) * 60

    if negative_delta_t_r.any():
        time_r[negative_delta_t_r] = T_t.squeeze(2)[:, num_drones:][negative_delta_t_r] + dis_r[
            negative_delta_t_r] / v_rmax
        battery_r[negative_delta_t_r] = battery.squeeze(2)[:, num_drones:][negative_delta_t_r] - (
                dis_r[negative_delta_t_r] / v_rmax * PC_r(current_demand[:, num_drones:][negative_delta_t_r],
                                                          v_rmax * torch.ones_like(vel_r)[
                                                              negative_delta_t_r] * 50 / 3)) * 60

    dynamic_battery = torch.cat((battery_d, battery_r), -1)
    dynamic_time = torch.cat((time_d, time_r), -1)

    if depot.any():
        dynamic_battery[:, :num_drones][depot[:, :num_drones]] = E[0]
        dynamic_battery[:, num_drones:][depot[:, num_drones:]] = E[1]
        vel_d = 0.8
        vel_r = 0.3
        dynamic_time[:, :num_drones][depot[:, :num_drones]] = dis_d[depot[:, :num_drones]] / vel_d + \
                                                              T_t.squeeze(2)[:, :num_drones][
                                                                  depot[:, :num_drones]] + 25
        dynamic_time[:, num_drones:][depot[:, num_drones:]] = dis_r[depot[:, num_drones:]] / vel_r + \
                                                              T_t.squeeze(2)[:, num_drones:][
                                                                  depot[:, num_drones:]] + 30
    # + torch.randint(35, 45,
    #                 (1,)).to(
    #     device)
    return dynamic_capacity.detach(), dynamic_time.detach(), dynamic_battery.detach()


def update_mask(demand, time_window, capacity, T_t, battery, num_drones, selected, mask, E, i):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, n_agents = selected.size()
    num_depots = torch.sum(demand == 0).item() // batch_size
    num_nodes = demand.size(1)
    num_pickup = (num_nodes - num_depots) // 2

    depot_indices = torch.arange(num_depots)
    pickup_indices = torch.arange(num_depots, num_depots + num_pickup)
    delivery_indices = torch.arange(num_depots + num_pickup, num_nodes)

    go_depot = selected.lt(num_depots)
    pickup_start = num_depots
    pickup_end = num_depots + num_pickup
    go_pickup_agent = selected.ge(pickup_start) & selected.lt(pickup_end)
    go_delivery_agent = selected.ge(num_depots + num_pickup) & selected.lt(num_nodes)

    # mask_ = mask.clone()
    mask = mask.scatter(2, selected.unsqueeze(-1), 1)
    maskfil = (mask.max(dim=1)[0])
    mask2 = mask.clone()
    maskfil = maskfil.unsqueeze(1).expand(batch_size, n_agents, num_nodes)

    if (~go_depot).any():
        # for depot in range(num_depots):
        #     mask2[(~go_depot).nonzero(as_tuple=True)[0], (~go_depot).nonzero(as_tuple=True)[1], depot] = 1
        mask2[(~go_depot).nonzero(as_tuple=True)[0], :, 0:num_depots] = 1

    corresponding_deliveries = num_depots + num_pickup + (pickup_indices - num_depots)

    mask[:, :, depot_indices] = torch.where(go_depot.unsqueeze(-1), 1, mask[:, :, depot_indices])

    # Adjusted operation for correct broadcasting
    unvisited_pickups = (mask2[:, :, pickup_indices] == 0)
    mask[:, :, pickup_indices] *= ~unvisited_pickups

    mask[:, :, delivery_indices] = 1
    # for p_idx, d_idx in zip(pickup_indices, corresponding_deliveries):
    mask[:, :, corresponding_deliveries] = torch.where(mask2[:, :, pickup_indices] == 1, 0,
                                                       mask[:, :, corresponding_deliveries])

    mask = torch.where(maskfil == 1, maskfil, mask)

    if i + 1 > demand.size(1):
        is_done = (mask2[:, :, num_depots:].sum(2) >= (demand.size(1) - num_depots)).float()
        combined = is_done.gt(num_depots)
        for depot in range(num_depots):
            mask2[combined.nonzero(as_tuple=True)[0], combined.nonzero(as_tuple=True)[1], depot] = 0

    threshold_values = torch.cat([torch.ones_like(battery[:, 0:num_drones]) * E[0] * 0.40,
                                  torch.ones_like(battery[:, num_drones:]) * E[1] * 0.25], 1)

    time_window_exceeds_battery = time_window.unsqueeze(1) > (torch.cat(
        [battery.squeeze(2)[:, 0:num_drones] / (60 * 1.5), battery.squeeze(2)[:, num_drones:] / (60 * 0.6)],
        -1).unsqueeze(2) + T_t)

    demand_exceeds_capacity = demand.unsqueeze(1) > capacity
    final_mask = demand_exceeds_capacity + mask2 + time_window_exceeds_battery + mask

    battery_exceeds_threshold = battery < threshold_values
    if battery_exceeds_threshold.any():
        # for agent in range(n_agents):
        final_mask[battery_exceeds_threshold.nonzero(as_tuple=True)[0], :, :num_depots] = 0
        final_mask[battery_exceeds_threshold.nonzero(as_tuple=True)[0], :, num_depots:num_nodes] = 1

    return final_mask.detach(), mask2.detach()
