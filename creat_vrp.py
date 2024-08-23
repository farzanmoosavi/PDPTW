import numpy as np
import torch
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fsolve
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
import torch
import random
from collections import defaultdict
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance_matrix


def generate_adjacency_network(node_data, time_windows, mode):
    # def generate_adjacency_network(G, selected_nodes, node_data, time_windows, mode):
    node_data_tensor = torch.tensor(node_data, dtype=torch.float32)
    time_windows_tensor = torch.tensor(time_windows, dtype=torch.float32)

    # Compute distance matrix using broadcasting

    dist_matrix = torch.norm(node_data_tensor.unsqueeze(1) - node_data_tensor, dim=2)

    # Set thresholds for connectivity
    if mode == 'drone':
        mu = random.randrange(1, 4)
        rho = random.randrange(30, 45)
    elif mode == 'robot':
        mu = random.randrange(1, 3)
        rho = random.randrange(20, 35)

    time_diff = torch.abs(time_windows_tensor.unsqueeze(0) - time_windows_tensor.unsqueeze(1)).squeeze(2)
    mask_adjacency = (dist_matrix < mu) & (time_diff <= rho)

    density = 0.2
    random_matrix = torch.rand(dist_matrix.shape)

    edge_adjustment = dist_matrix * random_matrix
    adjusted_edges = torch.where((random_matrix > density) & (mode == 'drone'),
                                 dist_matrix + edge_adjustment, dist_matrix)

    edges = torch.where(mask_adjacency, adjusted_edges, torch.zeros_like(dist_matrix))

    # Creating full indices grid
    n = node_data_tensor.size(0)
    rows, cols = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    edges_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)
    edges_index = torch.LongTensor(edges_index)
    # edges_index = edges_index.transpose(dim0=0, dim1=1)

    return edges.reshape(-1, 1), edges_index, mask_adjacency.reshape(-1, 1)


def create_instance(n_nodes, depots, n_robots, n_drones, random_seed=None):
    if random_seed is None:
        random_seed = np.random.randint(123456789)
    np.random.seed(random_seed)

    # N = np.random.randint(2 * n_nodes, 4 * n_nodes)
    # G = nx.gnm_random_graph(N, N * (N - 1) // 4)

    # selected_nodes = np.random.choice(list(G.nodes()), n_nodes + depots, replace=False)
    # positions = np.random.rand(N, 2) * 3  # Random positions scaled by 3
    # node_data = np.array([positions[node] for node in selected_nodes], dtype=np.float32)
    node_data = np.random.uniform(0, 5, (n_nodes + depots, 2))
    num_requests = n_nodes // 2
    # Define the rate λ (average number of requests in hours)
    lambda_rate = num_requests
    time_frame = 2 * 60  # in minutes

    # Generate the inter-arrival times (exponentially distributed)
    inter_arrival_times = np.random.exponential(scale=time_frame / lambda_rate, size=(num_requests, 1))
    arrival_times = np.cumsum(inter_arrival_times, axis=0)

    # Generate nodes with early and late time windows
    early_time = np.ceil(arrival_times)
    late_time = early_time + np.random.randint(35, 55, size=(num_requests, 1))
    time_window = np.vstack((np.zeros((depots, 1)), early_time, late_time))

    edges_d, edges_index_d, mask_adjacency_d = generate_adjacency_network(node_data, time_window,
                                                                          mode='drone')
    edges_r, edges_index_r, mask_adjacency_r = generate_adjacency_network(node_data, time_window,
                                                                          mode='robot')

    # edges_d, edges_index_d, mask_adjacency_d = generate_adjacency_network(G, selected_nodes, node_data, time_window,
    #                                                                       mode='drone')
    # edges_r, edges_index_r, mask_adjacency_r = generate_adjacency_network(G, selected_nodes, node_data, time_window,
    #                                                                       mode='robot')

    demand = np.random.randint(1, 10, size=(n_nodes // 2))
    demand = np.hstack((demand, -demand))  # Generating paired demand
    demand = np.insert(demand, 0, np.zeros(depots))  # Zero demand for depots

    CAPACITIES = {
        'drones': np.ones((n_drones)) * 5,
        'robots': np.ones((n_robots)) * 10
    }
    capacity = np.hstack((CAPACITIES['drones'], CAPACITIES['robots']))

    Battery = {
        'drones': torch.tensor(np.ones((n_drones)) * 7500),
        'robots': torch.tensor(np.ones((n_robots)) * 5500)
    }

    battery = torch.cat((Battery['drones'], Battery['robots']))

    return node_data, edges_d, edges_r, edges_index_d, time_window, demand, capacity, battery, mask_adjacency_d, mask_adjacency_r


def create_data(n_nodes, depots, n_robots, n_drones, num_samples, batch_size=32):
    dataset = []
    for _ in range(num_samples):
        node_data, edges_d, edges_r, edges_index, time_window, demand, capacity, battery, mask_adjacency_d, mask_adjacency_r = create_instance(
            n_nodes, depots, n_robots, n_drones)
        data = Data(
            x=torch.from_numpy(node_data).float(),
            edge_attr_d=edges_d.float(),
            edge_attr_r=edges_r.float(),
            edge_index=edges_index,
            demand=torch.tensor(demand).unsqueeze(-1).float(),
            capacity=torch.tensor(capacity).unsqueeze(-1).float(),
            battery=battery.unsqueeze(-1),
            time_window=torch.tensor(time_window).float(),
            mask_adjacency_d=mask_adjacency_d.float(),
            mask_adjacency_r=mask_adjacency_r.float()
        )

        dataset.append(data)

    return DataLoader(dataset, batch_size=batch_size)


def reward1(time_window, tour_indices, edge_attr_d, edge_attr_r, time, charge, num_drones):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, n_agent, steps = tour_indices.size()
    num_depots = torch.sum(time_window == 0).item() // batch_size
    num_nodes = torch.sum(time_window != 0).item() // batch_size
    num_pickup = num_nodes // 2
    time_window = time_window.view(batch_size, num_depots + num_nodes)  # （batch_size,nodes)
    edge_attr_d = edge_attr_d.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
    edge_attr_r = edge_attr_r.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)

    alpha, alpha1, alpha2, alpha3 = 1, 0.5, 0.4, 1.5
    charge = (charge - charge.min()) / (charge.max() - charge.min())
    total_cost = torch.zeros(batch_size, n_agent).to(device)

    for step in range(steps - 1):
        previous_indices = tour_indices[:, :, step]
        current_indices = tour_indices[:, :, step + 1]
        batch_d = torch.arange(batch_size).unsqueeze(1).expand(-1, num_drones)
        batch_r = torch.arange(batch_size).unsqueeze(1).expand(-1, n_agent - num_drones)

        # Calculate distances
        dis_d = edge_attr_d[batch_d, previous_indices[:, :num_drones], current_indices[:, :num_drones]]
        dis_r = edge_attr_r[batch_r, previous_indices[:, num_drones:], current_indices[:, num_drones:]]

        is_depot = current_indices < num_depots
        is_pickup = (current_indices >= num_depots) & (current_indices < num_depots + num_pickup)
        is_delivery = current_indices >= num_depots + num_pickup
        T_current = time[:, :, step]

        # Calculate penalties
        # current_time_window = torch.gather(time_window, 1, current_indices)
        pickup_t_window = torch.gather(time_window, 1, current_indices) * is_pickup
        # s_pickup = 1 + 2 * torch.rand(batch_size, n_agent)
        time_pickup = T_current * is_pickup

        delivery_t_window = torch.gather(time_window, 1, current_indices) * is_delivery
        # s_delivery = 1 + 2 * torch.rand(batch_size, n_agent)
        time_delivery = T_current * is_delivery

        penalty_pickup = torch.where(pickup_t_window - time_pickup > 0, torch.full_like(pickup_t_window, -5),
                                     alpha1 * (pickup_t_window - time_pickup))
        penalty_delivery = alpha2 * (time_delivery - delivery_t_window)

        if is_depot.any():
            penalty_pickup[is_depot] = 0
            penalty_delivery[is_depot] = 0
            T_current = torch.cat([dis_d / 0.8, dis_r / 0.3], -1)

        is_consecutive_depot = is_depot & (previous_indices < num_depots)
        T_current = torch.where(is_consecutive_depot, torch.zeros_like(T_current), T_current)
        penalty_pickup = torch.where(is_consecutive_depot, torch.zeros_like(penalty_pickup), penalty_pickup)
        penalty_delivery = torch.where(is_consecutive_depot, torch.zeros_like(penalty_delivery), penalty_delivery)

        total_cost += alpha * T_current + penalty_pickup + penalty_delivery

    # Apply penalty for negative battery values
    negative_battery_penalty = torch.where(charge < 0, charge * alpha3, torch.zeros_like(charge))
    total_cost += negative_battery_penalty.sum(dim=2)
    # mean = -torch.sum(total_cost, dim=1)
    reward = -total_cost

    return reward
