# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
# import math
# import numpy as np
# from torch.distributions.categorical import Categorical
# from vrpUpdate import update_mask, update_state
#
# import torch.utils.checkpoint as checkpoint
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# INIT = False
#
#
# def resolve_conflicts(p, index, num_depots):
#     # Create an expanded index view and compare all pairs within each batch
#     batch_size, num_agents = index.size()
#     expanded_index = index.unsqueeze(2)
#     comparison_matrix = expanded_index == expanded_index.transpose(1, 2)
#
#     # We don't want to compare each agent to itself, set diagonal to False
#     batch_range = torch.arange(batch_size)
#     agent_range = torch.arange(num_agents)
#     comparison_matrix[batch_range[:, None], agent_range, agent_range] = False
#
#     # Mask for valid conflicts (above the depot range)
#     valid_conflicts = (index >= num_depots).unsqueeze(2)
#
#     # Combine the conflicts with valid agent comparisons
#     conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)
#
#     # Now find conflicts and attempt to resolve them
#     while conflicts.any():
#         # Pick first valid conflict and resolve it (example approach)
#         conflict_positions = conflicts.nonzero(as_tuple=True)
#         batch_pos = conflict_positions[0][0]
#         agent_pos = conflict_positions[1][0]
#         conflicting_agent = conflict_positions[2][0]
#
#         # Invalidate the current choice
#         p[batch_pos, conflicting_agent, index[batch_pos, conflicting_agent]] = -float('inf')
#
#         # Resolve by choosing the next best
#         _, new_index = p[batch_pos, conflicting_agent].max(dim=-1)
#         index[batch_pos, conflicting_agent] = new_index
#
#         # Recalculate conflicts (could be optimized)
#         expanded_index = index.unsqueeze(2)
#         comparison_matrix = expanded_index == expanded_index.transpose(1, 2)
#         comparison_matrix[batch_range[:, None], agent_range, agent_range] = False
#         conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)
#
#     return index
#
#
# def resolve_conflicts_multinomial(p, index, non_depot_mask, num_depots):
#     batch_size, num_agents = index.size()
#     expanded_index = index.unsqueeze(2)
#     comparison_matrix = expanded_index == expanded_index.transpose(1, 2)
#
#     # Set diagonal to False to avoid self-comparison
#     batch_range = torch.arange(batch_size)
#     agent_range = torch.arange(num_agents)
#     comparison_matrix[batch_range[:, None], agent_range, agent_range] = False
#
#     # Mask for valid conflicts (above the depot range)
#     valid_conflicts = (index >= num_depots).unsqueeze(2)
#
#     # Combine the conflicts with valid agent comparisons
#     conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)
#
#     while conflicts.any():
#         # Pick the first valid conflict and resolve it
#         conflict_positions = conflicts.nonzero(as_tuple=True)
#         batch_pos = conflict_positions[0][0]
#         agent_pos = conflict_positions[1][0]
#         conflicting_agent = conflict_positions[2][0]
#
#         # Filter the probabilities with the non_depot_mask and set current depot index probabilities to zero
#         filtered_p = p[batch_pos, conflicting_agent] * (non_depot_mask[batch_pos, conflicting_agent] == 0).float()
#
#         # Ensure no zero probabilities are selected (set them to a very small positive value)
#         filtered_p[filtered_p == 0] = 1e-6
#
#         # Invalidate the current choice
#         filtered_p[index[batch_pos, conflicting_agent]] = 0
#
#         # Sample the next best index from the valid probabilities
#         new_index = torch.multinomial(filtered_p, 1).item()
#
#         # Update the index
#         index[batch_pos, conflicting_agent] = new_index
#
#         # Recalculate conflicts
#         expanded_index = index.unsqueeze(2)
#         comparison_matrix = expanded_index == expanded_index.transpose(1, 2)
#         comparison_matrix[batch_range[:, None], agent_range, agent_range] = False
#         conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)
#
#     return index
#
#
# class GatConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_channels,
#                  negative_slope=0.2, dropout=0):
#         super(GatConv, self).__init__(aggr='add')
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.Wvla = nn.Linear(in_channels, out_channels)
#         self.Wvld = nn.Linear(in_channels, out_channels)
#         self.Wvlp = nn.Linear(in_channels, out_channels)
#         self.Wgp = nn.Linear(2 * out_channels + edge_channels, out_channels)
#         self.Wgd = nn.Linear(2 * out_channels + edge_channels, out_channels)
#         self.Wga = nn.Linear(2 * out_channels + edge_channels, out_channels)
#
#     def forward(self, x, edge_index, edge_attr, num_depots, num_nodes, mask, batch_size, size=None):
#         # x = x.to(device)
#
#         x_x = x.view(batch_size, -1, self.out_channels)
#         # depot = x_x[:, 0:num_depots, :].reshape(batch_size * num_depots, self.out_channels)
#
#         pickup = x_x[:, num_depots:num_depots + num_nodes // 2, :].reshape(batch_size * num_nodes // 2,
#                                                                            self.out_channels)
#         delivery = x_x[:, num_depots + num_nodes // 2:num_depots + num_nodes, :].reshape(batch_size * num_nodes // 2,
#                                                                                          self.out_channels)
#
#         x_all = self.Wvla(x)
#
#         pickup = self.Wvlp(pickup)
#         delivery = self.Wvld(delivery)
#
#         # Reshape edge attributes and mask for each type of node
#         edge_attr = edge_attr.view(batch_size, num_nodes + num_depots, num_nodes + num_depots, -1)
#         edge_attr_pick = edge_attr[:, num_depots:num_depots + num_nodes // 2, num_depots:num_depots + num_nodes // 2,
#                          :].reshape(-1, edge_attr.size(-1))
#         edge_attr_del = edge_attr[:, num_depots + num_nodes // 2:num_depots + num_nodes,
#                         num_depots + num_nodes // 2:num_depots + num_nodes, :].reshape(-1, edge_attr.size(-1))
#         edge_attr = edge_attr.reshape(-1, edge_attr.size(-1))
#
#         mask = mask.view(batch_size, num_nodes + num_depots, num_nodes + num_depots)
#         mask_pick = mask[:, num_depots:num_depots + num_nodes // 2, num_depots:num_depots + num_nodes // 2].reshape(-1,
#                                                                                                                     1)
#         mask_del = mask[:, num_depots + num_nodes // 2:num_depots + num_nodes,
#                    num_depots + num_nodes // 2:num_depots + num_nodes].reshape(-1, 1)
#         mask = mask.reshape(-1, 1)
#
#         # Edge indices for pickup and delivery nodes
#         # subset_nodes = torch.arange(num_nodes // 2)
#         edge_index_pick = torch.stack([torch.arange(batch_size * num_nodes // 2).repeat_interleave(num_nodes // 2),
#                                        torch.arange(batch_size * num_nodes // 2).view(-1,
#                                                                                       num_nodes // 2).repeat_interleave(
#                                            num_nodes // 2, dim=0).flatten()]).to(device)
#         edge_index_del = edge_index_pick.clone()
#
#         X1 = self.propagate(edge_index, size=size, x=x_all, edge_attr=edge_attr, mask=mask, key='all')
#         X2 = self.propagate(edge_index_pick, size=size, x=pickup, edge_attr=edge_attr_pick, mask=mask_pick,
#                             key='pickup')
#         X3 = self.propagate(edge_index_del, size=size, x=delivery, edge_attr=edge_attr_del, mask=mask_del,
#                             key='delivery')
#
#         return torch.cat([X1, X2, X3], dim=0)
#         # return X1
#
#     def message(self, edge_index_i, x_i, x_j, size_i, edge_attr, mask, key):
#         x = torch.cat([x_i, x_j, edge_attr], dim=-1)
#         if key == 'all':
#             alpha = self.Wga(x)
#         elif key == 'pickup':
#             alpha = self.Wgp(x)
#         elif key == 'delivery':
#             alpha = self.Wgd(x)
#
#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         mask = mask.expand_as(alpha)
#         alpha = alpha.masked_fill(mask == 0, float('-inf'))
#         alpha = softmax(alpha, edge_index_i, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return x_j * alpha
#
#     def update(self, aggr_out):
#         return aggr_out
#
#
# # class GAT(nn.Module):
# #     def __init__(self, in_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0, nheads=4):
# #         super(GAT, self).__init__()
# #         self.dropout = dropout
# #         self.Wk = nn.Linear(out_channels * nheads, out_channels)
# #         self.attentions = nn.ModuleList([GatConv(in_channels, out_channels, edge_channels, dropout=dropout,
# #                                                  negative_slope=negative_slope) for _ in range(nheads)])
# #
# #     def forward(self, x, edge_index, edge_attr, num_depots, num_nodes, mask, batch_size):
# #         x = F.dropout(x, self.dropout, training=self.training)
# #         x = torch.cat(
# #             [att(x, edge_index, edge_attr, num_depots, num_nodes, mask, batch_size) for
# #              att in self.attentions], dim=1)
# #         x = self.Wk(x)
# #         return x
#
#
# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3):
#         super(Encoder, self).__init__()
#         self.hidden_node_dim = hidden_node_dim
#         # self.W0 = nn.Linear(input_dim, hidden_node_dim)
#         # self.b0 = nn.BatchNorm1d(hidden_node_dim)
#         self.W1 = nn.Linear(2 * input_dim, hidden_node_dim)
#         self.W2 = nn.Linear(input_dim, hidden_node_dim)
#         self.b1 = nn.BatchNorm1d(hidden_node_dim)
#         self.b2 = nn.BatchNorm1d(hidden_node_dim)
#         self.b3 = nn.BatchNorm1d(hidden_edge_dim)
#         # self.b4 = nn.BatchNorm1d(hidden_edge_dim)
#         self.W3 = nn.Linear(input_edge_dim, hidden_edge_dim)
#         # self.W4 = nn.Linear(input_edge_dim, hidden_edge_dim)
#
#         self.feedforward = nn.Sequential(
#             nn.Linear(hidden_node_dim, hidden_node_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_node_dim, hidden_node_dim)
#         )
#         self.batch_norm = nn.BatchNorm1d(hidden_node_dim)
#
#         self.convsd = nn.ModuleList(
#             [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])
#
#     def forward(self, data):
#         batch_size = data.num_graphs
#
#         dataset = torch.cat([data.x, data.demand, data.time_window], -1)
#         input_dim = dataset.shape[-1]
#         num_depots = torch.sum(data.demand == 0).item() // batch_size
#         num_nodes = torch.sum(data.demand != 0).item() // batch_size
#
#         dataset = dataset.view(batch_size, -1, input_dim)
#         emb_depot = dataset[:, 0:num_depots, :].reshape(batch_size * num_depots, input_dim)
#         emb_pickup = torch.cat([dataset[:, num_depots:num_depots + num_nodes // 2, :],
#                                 dataset[:, num_depots + num_nodes // 2:num_depots + num_nodes, :]], -1).reshape(
#             batch_size * num_nodes // 2, input_dim * 2)
#
#         emb_delivery = dataset[:, num_depots + num_nodes // 2:num_depots + num_nodes, :].reshape(
#             batch_size * num_nodes // 2, input_dim)
#
#         emb_depot = self.b2(self.W2(emb_depot)).view(batch_size, -1, self.hidden_node_dim)
#
#         emb_pickup = self.b1(self.W1(emb_pickup)).view(batch_size, -1, self.hidden_node_dim)
#
#         emb_delivery = self.b2(self.W2(emb_delivery)).view(batch_size, -1, self.hidden_node_dim)
#
#         edge_attr_d = self.b3(self.W3(data.edge_attr_d))
#         edge_attr_r = self.b3(self.W3(data.edge_attr_r))
#
#         x_d = torch.cat([emb_depot, emb_pickup, emb_delivery], 1).view(-1, self.hidden_node_dim)
#         x_r = x_d
#         mask = data.mask_adjacency_d
#         maskr = data.mask_adjacency_r
#         for conv in self.convsd:
#             Xd = conv(x_d, data.edge_index, edge_attr_d, num_depots, num_nodes, mask, batch_size)
#             Xr = conv(x_r, data.edge_index, edge_attr_r, num_depots, num_nodes, maskr, batch_size)
#
#             Xd = Xd.view(batch_size, -1, Xd.size(1))
#             Xr = Xr.view(batch_size, -1, Xr.size(1))
#
#             xalld, xpickd, xdeld = Xd[:, 0:num_depots + num_nodes, :], Xd[:,
#                                                                        num_depots + num_nodes:num_depots + 3 * num_nodes // 2,
#                                                                        :], Xd[:,
#                                                                            num_depots + 3 * num_nodes // 2:num_depots + 2 * num_nodes,
#                                                                            :]
#             xallr, xpickr, xdelr = Xr[:, 0:num_depots + num_nodes, :], Xr[:,
#                                                                        num_depots + num_nodes:num_depots + 3 * num_nodes // 2,
#                                                                        :], Xr[:,
#                                                                            num_depots + 3 * num_nodes // 2:num_depots + 2 * num_nodes,
#                                                                            :]
#
#             Xd = torch.cat([xalld[:, :num_depots, :],
#                             xalld[:, num_depots:num_depots + num_nodes // 2, :] + xpickd,
#                             xalld[:, num_depots + num_nodes // 2:num_depots + num_nodes, :] + xdeld], 1).view(-1,
#                                                                                                               self.hidden_node_dim)
#             Xr = torch.cat([xallr[:, :num_depots, :],
#                             xallr[:, num_depots:num_depots + num_nodes // 2, :] + xpickr,
#                             xallr[:, num_depots + num_nodes // 2:num_depots + num_nodes, :] + xdelr], 1).view(-1,
#                                                                                                               self.hidden_node_dim)
#
#             x_d = Xd
#             x_r = Xr
#
#         xd = self.batch_norm(x_d + self.feedforward(x_d))
#         xr = self.batch_norm(x_r + self.feedforward(x_r))
#
#         xd = xd.reshape((batch_size, -1, self.hidden_node_dim))
#         xr = xr.reshape((batch_size, -1, self.hidden_node_dim))
#
#         return xd, xr
#
#
# class Attention1(nn.Module):
#     def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
#         super(Attention1, self).__init__()
#
#         self.n_heads = n_heads
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.head_dim = hidden_dim // n_heads
#         self.norm = 1 / math.sqrt(self.head_dim)
#
#         self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
#         self.k = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.v = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
#
#         if INIT:
#             for name, p in self.named_parameters():
#                 if 'weight' in name:
#                     if len(p.size()) >= 2:
#                         nn.init.orthogonal_(p, gain=1)
#                 elif 'bias' in name:
#                     nn.init.constant_(p, 0)
#
#     def forward(self, state_t, context, mask):
#         '''
#         :param state_t: (batch_size, num_agents, input_dim * cat)
#         :param context: (batch_size, n_nodes, input_dim)
#         :param mask: selected nodes (batch_size, n_nodes)
#         :return:
#         '''
#         batch_size, num_agents, _ = state_t.size()
#         n_nodes = context.size(1)
#         input_dim = self.input_dim
#
#         Q = self.w(state_t).view(batch_size, num_agents, self.n_heads, self.head_dim)
#         K = self.k(context).view(batch_size, n_nodes, self.n_heads, self.head_dim)
#         V = self.v(context).view(batch_size, n_nodes, self.n_heads, self.head_dim)
#
#         Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
#
#         compatibility = self.norm * torch.matmul(Q, K.transpose(2,
#                                                                 3))  # (batch_size, n_heads, num_agents, head_dim) * (batch_size, n_heads, head_dim, n_nodes)
#         # compatibility = compatibility.squeeze(2)  # (batch_size, n_heads, num_agents, n_nodes)
#
#         mask = mask.unsqueeze(1).expand_as(compatibility)
#         u_i = compatibility.masked_fill(mask.bool(), float(-10000))
#
#         scores = F.softmax(u_i, dim=-1)  # (batch_size, n_heads, num_agents, n_nodes)
#         scores = scores.unsqueeze(3)
#
#         out_put = torch.matmul(scores, V.unsqueeze(
#             2))  # (batch_size, n_heads, num_agents, 1, n_nodes) * (batch_size, n_heads, 1, n_nodes, head_dim)
#         out_put = out_put.squeeze(3).view(batch_size, num_agents,
#                                           self.hidden_dim)  # (batch_size, num_agents, hidden_dim)
#
#         out_put = self.fc(out_put)
#         return out_put  # (batch_size, num_agents, hidden_dim)
#
#
# class ProbAttention(nn.Module):
#     def __init__(self, n_heads, input_dim, hidden_dim):
#         super(ProbAttention, self).__init__()
#         self.n_heads = n_heads
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#
#         self.norm = 1 / math.sqrt(hidden_dim)
#         self.k = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.mhalayer = Attention1(n_heads, 1, input_dim, hidden_dim)
#
#         if INIT:
#             for name, p in self.named_parameters():
#                 if 'weight' in name:
#                     if len(p.size()) >= 2:
#                         nn.init.orthogonal_(p, gain=1)
#                 elif 'bias' in name:
#                     nn.init.constant_(p, 0)
#
#     def forward(self, state_t, context, context_d, n_d, mask, T):
#         '''
#         :param state_t: (batch_size, num_agents, input_dim * 3)
#         :param context: (batch_size, n_nodes, input_dim)
#         :param mask: selected nodes (batch_size, num_agents, n_nodes)
#         :return: softmax_score
#         '''
#         batch_size, num_agents, _ = state_t.size()
#         x_d = self.mhalayer(state_t[:, 0:n_d, :], context_d, mask[:, 0:n_d, :])
#         x_r = self.mhalayer(state_t[:, n_d:num_agents, :], context, mask[:, n_d:num_agents, :])
#         x = torch.cat([x_d, x_r], 1)
#         n_nodes = context.size(1)
#
#         Q = x.view(batch_size, num_agents, -1)
#         K = self.k(context).view(batch_size, n_nodes, -1)
#
#         compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))  # (batch_size, num_agents, n_nodes)
#
#         mask = mask.expand_as(compatibility)
#         compatibility = compatibility.masked_fill(mask.bool(), float(-10000))
#
#         scores = F.softmax(compatibility / T, dim=-1)  # (batch_size, num_agents, n_nodes)
#         return scores
#
#
# class Decoder1(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(Decoder1, self).__init__()
#
#         super(Decoder1, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#
#         self.prob = ProbAttention(8, input_dim, hidden_dim)
#
#         self.fcr = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         # self.fc1r = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.fcd = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         # self.fc1d = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.Wd = nn.Linear(hidden_dim + 3, hidden_dim, bias=False)
#         self.Wr = nn.Linear(hidden_dim + 3, hidden_dim, bias=False)
#         # self.bd = nn.BatchNorm1d(hidden_dim)
#         # self.br = nn.BatchNorm1d(hidden_dim)
#         if INIT:
#             for name, p in self.named_parameters():
#                 if 'weight' in name:
#                     if len(p.size()) >= 2:
#                         nn.init.orthogonal_(p, gain=1)
#                 elif 'bias' in name:
#                     nn.init.constant_(p, 0)
#
#     def forward(self, emb_d, emb_r, pool_d, pool_r, capacity, demand, battery, time_window, num_depots, num_nodes,
#                 num_drones, num_robots, edge_attr_d, edge_attr_r, T, greedy=False):
#         batch_size, _, hidden = emb_d.size()
#         mask1 = emb_d.new_zeros((batch_size, num_drones + num_robots, emb_d.size(1)))
#         mask = emb_d.new_zeros((batch_size, num_drones + num_robots, emb_d.size(1)))
#
#         battery = battery.view(batch_size, num_drones + num_robots).unsqueeze(2).float().to(device)
#         capacity = capacity.view(batch_size, num_drones + num_robots).unsqueeze(2)
#         # U_t = torch.zeros(batch_size, num_drones + num_robots).unsqueeze(2)  # load
#         T_t = torch.zeros(batch_size, num_drones + num_robots).unsqueeze(2).float().to(device)  # time
#         demands = demand.view(batch_size, emb_d.size(1))  # (batch_size, nodes)
#         time_window = time_window.view(batch_size, emb_d.size(1))  # (batch_size, nodes)
#         edge_attr_d = edge_attr_d.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
#         edge_attr_r = edge_attr_r.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
#         index = torch.zeros(batch_size, num_drones + num_robots).to(device).long()
#
#         E_d = battery[0, 0].item()
#         C_d = capacity[0, 0].item()
#         E_r = battery[0, num_drones + num_robots - 1].item()
#         C_r = capacity[0, num_drones + num_robots - 1].item()
#         E = [E_d, E_r]
#         C = [C_d, C_r]
#
#         log_ps = []
#         actions = []
#         time_log = []
#         BL = []
#
#         stepsize = 0
#
#         while (mask1[:, :, num_depots:].max(dim=1)[0]).eq(0).any():
#
#             if not (mask1[:, :, num_depots:].max(dim=1)[0]).eq(0).any():
#                 break
#
#             context = []
#             decoder_input = []
#             for num in range(num_drones + num_robots):
#
#                 if stepsize == 0:
#                     depot = np.random.randint(0, num_depots, 1)[0]
#                     s_t_i = emb_d[:, depot, :]
#                     index[:, num] = depot
#
#
#                 else:
#                     s_t_i = s_t[:, num, :]
#
#                 if num < num_drones:
#                     context.append((self.Wd(
#                         torch.cat([s_t_i, capacity[:, num], T_t[:, num], battery[:, num]], -1))).unsqueeze(1))
#
#                 else:
#                     context.append((self.Wr(
#                         torch.cat([s_t_i, capacity[:, num], T_t[:, num], battery[:, num]], -1)
#
#                     )).unsqueeze(1))
#
#             context = torch.cat(context, dim=1)
#
#             if stepsize == 0:
#                 actions.append(index.data.unsqueeze(2))
#             # input_d = torch.cat(
#             #     [pool_d.unsqueeze(1).expand(emb_d.size(0), num_drones + num_robots, pool_d.size(1)), context], -1)
#             #
#             # input_r = torch.cat(
#             #     [pool_r.unsqueeze(1).expand(emb_d.size(0), num_drones + num_robots, pool_r.size(1)), context], -1)
#             #
#             # veh_d_mean = (self.fcd(input_d)).mean(dim=1)
#             # veh_r_mean = (self.fcd(input_r)).mean(dim=1)
#             veh_d_mean = pool_d + (self.fcd(context[:, :num_drones, :])).mean(dim=1)
#             veh_r_mean = pool_r + (self.fcr(context[:, num_drones:, :])).mean(dim=1)
#
#             for num in range(num_drones + num_robots):
#
#                 if battery[0, num].item() == E_d:
#                     # _input = (veh_d_mean + self.fc1d(context[:, num, :])).unsqueeze(1)
#
#                     _input = (veh_d_mean + context[:, num, :]).unsqueeze(1)
#
#                 elif battery[0, num].item() == E_r:
#                     # _input = (veh_r_mean + self.fc1r(context[:, num, :])).unsqueeze(1)
#                     _input = (veh_r_mean + context[:, num, :]).unsqueeze(1)
#
#                 decoder_input.append(_input)
#
#             decoder_input = torch.cat(decoder_input, dim=1)
#
#             if stepsize == 0:
#                 mask, mask1 = update_mask(demands, time_window, capacity, T_t, battery, num_drones, index, mask1, E,
#                                           stepsize)
#
#             p = self.prob(decoder_input, emb_r, emb_d, num_drones, mask, T)
#             p = torch.where(torch.isnan(p) | torch.isinf(p), torch.tensor(0.0001, dtype=p.dtype), p)
#             # dist = Categorical(p)
#             if greedy:
#                 _, index = p.max(dim=-1)
#                 index = resolve_conflicts(p, index, num_depots)
#             else:
#                 non_depot_mask = (mask1.max(dim=1)[0] == 1).unsqueeze(1).expand_as(mask)
#                 p = p.masked_fill(non_depot_mask, 0.0001)
#                 index = torch.multinomial(p.view(-1, p.size(-1)), 1).view(batch_size, num_drones + num_robots)
#                 index = resolve_conflicts_multinomial(p, index, non_depot_mask, num_depots)
#                 # dist = Categorical(p)
#
#             actions.append(index.data.unsqueeze(2))
#             # log_p = dist.log_prob(index)
#             batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, num_drones + num_robots)
#             agent_indices = torch.arange(num_drones + num_robots).unsqueeze(0).expand(batch_size,
#                                                                                       num_drones + num_robots)
#
#             log_p = torch.log(p[batch_indices, agent_indices, index] + 1e-2)
#
#             is_done = (mask1[:, :, num_depots:].max(dim=1)[0].sum(1).unsqueeze(1).expand(batch_size,
#                                                                                          num_drones + num_robots) >= (
#                                emb_d.size(1) - num_depots)).float()
#             log_p = log_p * (1. - is_done)
#
#             log_ps.append(log_p.unsqueeze(2))
#
#             capacity, T_t, battery = update_state(demands, time_window, battery, T_t, capacity, index, E, C, num_drones,
#                                                   actions, edge_attr_r, edge_attr_d, stepsize)
#
#             capacity, T_t, battery = capacity.unsqueeze(2), T_t.unsqueeze(2), battery.unsqueeze(2)
#             time_log.append(T_t)
#             BL.append(battery)
#             mask, mask1 = update_mask(demands, time_window, capacity, T_t, battery, num_drones, index, mask1, E,
#                                       stepsize)
#
#             s_t = []
#             for num in range(num_drones + num_robots):
#                 if num < num_drones:
#                     _input = torch.gather(emb_d, 1, index[:, num].unsqueeze(1).unsqueeze(2).expand(emb_d.size(0), -1,
#                                                                                                    emb_d.size(2)))
#                 else:
#                     _input = torch.gather(emb_r, 1, index[:, num].unsqueeze(1).unsqueeze(2).expand(emb_r.size(0), -1,
#                                                                                                    emb_r.size(2)))
#                 s_t.append(_input)
#
#             s_t = torch.cat(s_t, dim=1)
#
#             stepsize += 1
#
#         log_ps = torch.cat(log_ps, dim=2)
#         time_log = torch.cat(time_log, dim=2)
#         BL = torch.cat(BL, dim=2)
#         actions = torch.cat(actions, dim=2)
#         # print(actions[0:10])
#         log_p = log_ps.sum(dim=2)
#
#         return actions, log_p, time_log, BL
#
#
# class Model(nn.Module):
#     def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
#         super(Model, self).__init__()
#         self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
#         self.decoder = Decoder1(hidden_node_dim, hidden_node_dim)
#
#     def forward(self, datas, num_drones, num_robots, greedy=False, T=1):
#         x_d, x_r = self.encoder(datas)  # (batch,seq_len,hidden_node_dim)
#
#         pooledx = x_d.mean(dim=1)
#         pooledr = x_r.mean(dim=1)
#         batch_size = datas.num_graphs
#
#         num_depots = torch.sum(datas.demand == 0).item() // batch_size
#         num_nodes = torch.sum(datas.demand != 0).item() // batch_size
#         demand = datas.demand
#         time_window = datas.time_window
#         capacity = datas.capacity
#         battery = datas.battery
#         edge_attr_d = datas.edge_attr_d
#         edge_attr_r = datas.edge_attr_r
#
#         actions, log_p, time, BL = self.decoder(x_d, x_r, pooledx, pooledr, capacity, demand, battery, time_window,
#                                                 num_depots, num_nodes, num_drones, num_robots, edge_attr_d, edge_attr_r,
#                                                 T, greedy)
#         return actions, log_p, time, BL


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.utils.checkpoint as checkpoint
from typing import NamedTuple
import math
# from torch.amp import autocast

import numpy as np
from torch.distributions.categorical import Categorical
from vrpUpdate import update_mask, update_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INIT = False


def resolve_conflicts(p, index, num_depots):
    # Create an expanded index view and compare all pairs within each batch
    batch_size, num_agents = index.size()
    expanded_index = index.unsqueeze(2)
    comparison_matrix = expanded_index == expanded_index.transpose(1, 2)

    # We don't want to compare each agent to itself, set diagonal to False
    batch_range = torch.arange(batch_size)
    agent_range = torch.arange(num_agents)
    comparison_matrix[batch_range[:, None], agent_range, agent_range] = False

    # Mask for valid conflicts (above the depot range)
    valid_conflicts = (index >= num_depots).unsqueeze(2)

    # Combine the conflicts with valid agent comparisons
    conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)

    # Now find conflicts and attempt to resolve them
    while conflicts.any():
        # Pick first valid conflict and resolve it (example approach)
        conflict_positions = conflicts.nonzero(as_tuple=True)
        batch_pos = conflict_positions[0][0]
        agent_pos = conflict_positions[1][0]
        conflicting_agent = conflict_positions[2][0]

        # Invalidate the current choice
        p[batch_pos, conflicting_agent, index[batch_pos, conflicting_agent]] = -float('inf')

        # Resolve by choosing the next best
        _, new_index = p[batch_pos, conflicting_agent].max(dim=-1)
        index[batch_pos, conflicting_agent] = new_index

        # Recalculate conflicts (could be optimized)
        expanded_index = index.unsqueeze(2)
        comparison_matrix = expanded_index == expanded_index.transpose(1, 2)
        comparison_matrix[batch_range[:, None], agent_range, agent_range] = False
        conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)

    return index


def resolve_conflicts_multinomial(p, index, non_depot_mask, num_depots):
    batch_size, num_agents = index.size()
    expanded_index = index.unsqueeze(2)
    comparison_matrix = expanded_index == expanded_index.transpose(1, 2)

    # Set diagonal to False to avoid self-comparison
    batch_range = torch.arange(batch_size)
    agent_range = torch.arange(num_agents)
    comparison_matrix[batch_range[:, None], agent_range, agent_range] = False

    # Mask for valid conflicts (above the depot range)
    valid_conflicts = (index >= num_depots).unsqueeze(2)

    # Combine the conflicts with valid agent comparisons
    conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)

    while conflicts.any():
        # Pick the first valid conflict and resolve it
        conflict_positions = conflicts.nonzero(as_tuple=True)
        batch_pos = conflict_positions[0][0]
        agent_pos = conflict_positions[1][0]
        conflicting_agent = conflict_positions[2][0]

        # Filter the probabilities with the non_depot_mask and set current depot index probabilities to zero
        filtered_p = p[batch_pos, conflicting_agent] * (non_depot_mask[batch_pos, conflicting_agent] == 0).float()

        # Ensure no zero probabilities are selected (set them to a very small positive value)
        filtered_p[filtered_p == 0] = 1e-6

        # Invalidate the current choice
        filtered_p[index[batch_pos, conflicting_agent]] = 0

        # Sample the next best index from the valid probabilities
        new_index = torch.multinomial(filtered_p, 1).item()

        # Update the index
        index[batch_pos, conflicting_agent] = new_index

        # Recalculate conflicts
        expanded_index = index.unsqueeze(2)
        comparison_matrix = expanded_index == expanded_index.transpose(1, 2)
        comparison_matrix[batch_range[:, None], agent_range, agent_range] = False
        conflicts = comparison_matrix & valid_conflicts & valid_conflicts.transpose(1, 2)

    return index


class GatConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,
                 negative_slope=0.2, dropout=0):
        super(GatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.Wvla = nn.Linear(in_channels, out_channels, bias=False)
        self.Wvld = nn.Linear(in_channels, out_channels, bias=False)
        self.Wvlp = nn.Linear(in_channels, out_channels, bias=False)
        self.Wgp = nn.Linear(2 * out_channels + edge_channels, out_channels, bias=False)
        self.Wgd = nn.Linear(2 * out_channels + edge_channels, out_channels, bias=False)
        self.Wga = nn.Linear(2 * out_channels + edge_channels, out_channels, bias=False)

    # @autocast(device_type='cuda')
    def forward(self, x, edge_index, edge_attr, num_depots, num_nodes, mask, batch_size, size=None):
        # x = x
        # edge_index = edge_index
        # edge_attr = edge_attr
        # mask = mask

        x_x = x.view(batch_size, -1, self.out_channels)
        # depot = x_x[:, 0:num_depots, :].reshape(batch_size * num_depots, self.out_channels)

        pickup = x_x[:, num_depots:num_depots + num_nodes // 2, :].reshape(batch_size * num_nodes // 2,
                                                                           self.out_channels)
        delivery = x_x[:, num_depots + num_nodes // 2:num_depots + num_nodes, :].reshape(batch_size * num_nodes // 2,
                                                                                         self.out_channels)

        x_all = self.Wvla(x)

        pickup = self.Wvlp(pickup)
        delivery = self.Wvld(delivery)

        # edge_attr = edge_attr.view(batch_size, num_nodes + num_depots, num_nodes + num_depots, edge_attr.size(1))
        # edge_attr_pick = edge_attr[:, num_depots:num_depots + num_nodes // 2, num_depots:num_depots + num_nodes // 2, :].reshape(batch_size * (num_nodes // 2) ** 2, edge_attr.size(3))
        # edge_attr_del = edge_attr[:, num_depots + num_nodes // 2:num_depots + num_nodes, num_depots + num_nodes // 2:num_depots + num_nodes, :].reshape(batch_size * (num_nodes // 2) ** 2, edge_attr.size(3))
        # edge_attr = edge_attr.reshape(batch_size * (num_nodes + num_depots) ** 2, edge_attr.size(3))

        # mask = mask.view(batch_size, num_nodes + num_depots, num_nodes + num_depots)
        # mask_pick = mask[:, num_depots:num_depots + num_nodes // 2, num_depots:num_depots + num_nodes // 2].reshape(batch_size * (num_nodes // 2) ** 2, 1)
        # mask_del = mask[:, num_depots + num_nodes // 2:num_depots + num_nodes, num_depots + num_nodes // 2:num_depots + num_nodes].reshape(batch_size * (num_nodes // 2) ** 2, 1)
        # mask = mask.reshape(batch_size * (num_nodes + num_depots) ** 2, 1)

        # subset_nodes = torch.arange(num_nodes // 2)
        # first_dim = torch.arange(batch_size * num_nodes // 2).repeat_interleave(num_nodes // 2)
        # second_dim = torch.arange(batch_size * num_nodes // 2)
        # groups = second_dim.view(-1, num_nodes // 2)
        # second_dim = groups.repeat_interleave(num_nodes // 2, dim=0).flatten()
        # edge_index_pick = torch.stack([first_dim, second_dim])
        # edge_index_del = edge_index_pick

        # attention_weights = {'all': [edge_index, x_all, edge_attr, mask]}  #,
        #
        # # 'pickup': [edge_index_pick, pickup, edge_attr_pick, mask_pick],
        # # 'delivery': [edge_index_del, delivery, edge_attr_del, mask_del]}
        #
        # for key, val in attention_weights.items():
        #     if key == 'all':
        #         X1 = self.propagate(val[0], size=size, x=val[1], edge_attr=val[2], mask=val[3], key=key)
        # #     elif key == 'pickup':
        # #         X2 = self.propagate(val[0], size=size, x=val[1], edge_attr=val[2], mask=val[3], key=key)
        # #     elif key == 'delivery':
        # #         X3 = self.propagate(val[0], size=size, x=val[1], edge_attr=val[2], mask=val[3], key=key)

        # Reshape edge attributes and mask for each type of node
        edge_attr = edge_attr.view(batch_size, num_nodes + num_depots, num_nodes + num_depots, -1)
        edge_attr_pick = edge_attr[:, num_depots:num_depots + num_nodes // 2, num_depots:num_depots + num_nodes // 2,
                         :].reshape(-1, edge_attr.size(-1))
        edge_attr_del = edge_attr[:, num_depots + num_nodes // 2:num_depots + num_nodes,
                        num_depots + num_nodes // 2:num_depots + num_nodes, :].reshape(-1, edge_attr.size(-1))
        edge_attr = edge_attr.reshape(-1, edge_attr.size(-1))

        mask = mask.view(batch_size, num_nodes + num_depots, num_nodes + num_depots)
        mask_pick = mask[:, num_depots:num_depots + num_nodes // 2, num_depots:num_depots + num_nodes // 2].reshape(-1,
                                                                                                                    1)
        mask_del = mask[:, num_depots + num_nodes // 2:num_depots + num_nodes,
                   num_depots + num_nodes // 2:num_depots + num_nodes].reshape(-1, 1)
        mask = mask.reshape(-1, 1)

        # Edge indices for pickup and delivery nodes
        # subset_nodes = torch.arange(num_nodes // 2)
        edge_index_pick = torch.stack([torch.arange(batch_size * num_nodes // 2).repeat_interleave(num_nodes // 2),
                                       torch.arange(batch_size * num_nodes // 2).view(-1,
                                                                                      num_nodes // 2).repeat_interleave(
                                           num_nodes // 2, dim=0).flatten()]).to(device)
        edge_index_del = edge_index_pick.clone()

        # Compute attention weights and propagate messages

        X1 = self.propagate(edge_index, size=size, x=x_all, edge_attr=edge_attr, mask=mask, key='all')
        X2 = self.propagate(edge_index_pick, size=size, x=pickup, edge_attr=edge_attr_pick, mask=mask_pick,
                            key='pickup')
        X3 = self.propagate(edge_index_del, size=size, x=delivery, edge_attr=edge_attr_del, mask=mask_del,
                            key='delivery')

        # return self.propagate(edge_index, size=size, x=x_all, edge_attr=edge_attr, mask=mask, key='all')
        return torch.cat([X1, X2, X3], dim=0)
        # return X1

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr, mask, key):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        if key == 'all':
            alpha = self.Wga(x)
        elif key == 'pickup':
            alpha = self.Wgp(x)
        elif key == 'delivery':
            alpha = self.Wgd(x)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        mask = mask.expand_as(alpha)
        alpha = alpha.masked_fill(mask == 0, float('-inf'))
        alpha = softmax(alpha, edge_index_i, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


# class GAT(nn.Module):
#     def __init__(self, in_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0, nheads=4):
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.Wk = nn.Linear(out_channels * nheads, out_channels)
#         self.attentions = nn.ModuleList(
#             [GatConv(in_channels, out_channels, edge_channels, dropout=dropout, negative_slope=negative_slope) for _ in
#              range(nheads)])
#
#     def forward(self, x, edge_index, edge_attr, num_depots, num_nodes, mask, batch_size):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat(
#             [att(x, edge_index, edge_attr, num_depots, num_nodes, mask, batch_size) for att in self.attentions], dim=1)
#         x = self.Wk(x)
#         return x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.W0 = nn.Linear(input_dim, hidden_node_dim, bias=False)
        self.b0 = nn.BatchNorm1d(hidden_node_dim)
        self.W1 = nn.Linear(2 * input_dim, hidden_node_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_node_dim, bias=False)
        self.b1 = nn.BatchNorm1d(hidden_node_dim)
        self.b2 = nn.BatchNorm1d(hidden_node_dim)
        self.b3 = nn.BatchNorm1d(hidden_edge_dim)
        self.b4 = nn.BatchNorm1d(hidden_edge_dim)
        # self.b5 = nn.BatchNorm1d(hidden_node_dim)
        self.W3 = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)
        self.W4 = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim),
            nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim)
        )
        self.batch_norm = nn.BatchNorm1d(hidden_node_dim)

        self.convsd = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])
        # self.convsr = nn.ModuleList(
        #     [GAT(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])

    def forward(self, data):
        batch_size = data['x'].shape[0]

        # Access tensors from the dictionary
        dataset = torch.cat([data['x'], data['demand'], data['time_window']], -1)
        input_dim = dataset.shape[-1]
        num_depots = torch.sum(data['demand'] == 0).item() // batch_size
        num_nodes = torch.sum(data['demand'] != 0).item() // batch_size

        dataset = dataset.view(batch_size, -1, input_dim)
        emb_depot = dataset[:, 0:num_depots, :].reshape(batch_size * num_depots, input_dim)
        emb_pickup = torch.cat([dataset[:, num_depots:num_depots + num_nodes // 2, :],
                                dataset[:, num_depots + num_nodes // 2:num_depots + num_nodes, :]], -1).reshape(
            batch_size * num_nodes // 2, input_dim * 2)

        emb_delivery = dataset[:, num_depots + num_nodes // 2:num_depots + num_nodes, :].reshape(
            batch_size * num_nodes // 2, input_dim)

        emb_depot = self.b0(self.W0(emb_depot)).view(batch_size, -1, self.hidden_node_dim)

        emb_pickup = self.b1(self.W1(emb_pickup)).view(batch_size, -1, self.hidden_node_dim)

        emb_delivery = self.b2(self.W2(emb_delivery)).view(batch_size, -1, self.hidden_node_dim)

        # edge_attr_d = self.b3(self.W3(data['edge_attr_d']))
        # edge_attr_r = self.b4(self.W4(data['edge_attr_r']))

        edge_attr_d = data['edge_attr_d'].view(-1, data['edge_attr_d'].shape[-1])
        edge_attr_r = data['edge_attr_r'].view(-1, data['edge_attr_r'].shape[-1])
        edge_index = data['edge_index'].view(data['edge_index'].shape[1], -1)

        # Apply the BatchNorm1d layer correctly based on the reshaped tensor dimensions
        edge_attr_d = self.b3(self.W3(edge_attr_d)).view(-1, self.hidden_edge_dim)
        edge_attr_r = self.b4(self.W4(edge_attr_r)).view(-1, self.hidden_edge_dim)

        x_d = torch.cat([emb_depot, emb_pickup, emb_delivery], 1).view(-1, self.hidden_node_dim)
        x_r = x_d
        mask = data['mask_adjacency_d']
        maskr = data['mask_adjacency_r']
        for conv in self.convsd:
            Xd = conv(x_d, edge_index, edge_attr_d, num_depots, num_nodes, mask, batch_size)
            Xr = conv(x_r, edge_index, edge_attr_r, num_depots, num_nodes, maskr, batch_size)

            Xd = Xd.view(batch_size, -1, Xd.size(1))
            Xr = Xr.view(batch_size, -1, Xr.size(1))

            xalld, xpickd, xdeld = Xd[:, 0:num_depots + num_nodes, :], Xd[:,
                                                                       num_depots + num_nodes:num_depots + 3 * num_nodes // 2,
                                                                       :], Xd[:,
                                                                           num_depots + 3 * num_nodes // 2:num_depots + 2 * num_nodes,
                                                                           :]
            xallr, xpickr, xdelr = Xr[:, 0:num_depots + num_nodes, :], Xr[:,
                                                                       num_depots + num_nodes:num_depots + 3 * num_nodes // 2,
                                                                       :], Xr[:,
                                                                           num_depots + 3 * num_nodes // 2:num_depots + 2 * num_nodes,
                                                                           :]

            # xpicknewd = xalld[:, num_depots:num_depots + num_nodes // 2, :] + xpickd
            # xdelnewd = xalld[:, num_depots + num_nodes // 2:num_depots + num_nodes, :] + xdeld
            # emb_depot = xalld[:, 0:num_depots, :]
            # x1 = torch.cat([emb_depot, xpicknewd, xdelnewd], 1)
            # x1 = x1.view(-1, self.hidden_node_dim)
            # x_d = x_d + x1
            # x_d = self.b5(x_d)
            # x_d = self.batch_norm((x_d + self.feedforward(x_d)))
            Xd = torch.cat([xalld[:, :num_depots, :],
                            xalld[:, num_depots:num_depots + num_nodes // 2, :] + xpickd,
                            xalld[:, num_depots + num_nodes // 2:num_depots + num_nodes, :] + xdeld], 1).view(-1,
                                                                                                              self.hidden_node_dim)
            Xr = torch.cat([xallr[:, :num_depots, :],
                            xallr[:, num_depots:num_depots + num_nodes // 2, :] + xpickr,
                            xallr[:, num_depots + num_nodes // 2:num_depots + num_nodes, :] + xdelr], 1).view(-1,
                                                                                                              self.hidden_node_dim)

            x_d = Xd
            x_r = Xr

        xd = self.batch_norm(x_d + self.feedforward(x_d))
        xr = self.batch_norm(x_r + self.feedforward(x_r))

        xd = xd.reshape((batch_size, -1, self.hidden_node_dim))
        xr = xr.reshape((batch_size, -1, self.hidden_node_dim))

        # for conv in self.convsr:
        #     mask = data.mask_adjacency_r
        #     X = conv(x, data.edge_index, edge_attr_r, num_depots, num_nodes, mask, batch_size)
        #     X = X.view(batch_size, -1, X.size(1))
        #     # xall, xpick, xdel = X[:, 0:num_depots + num_nodes, :], X[:, num_depots + num_nodes:num_depots + 3 * num_nodes // 2, :], X[:, num_depots + 3 * num_nodes // 2:num_depots + 2 * num_nodes, :]
        #     # xpicknew = xall[:, num_depots:num_depots + num_nodes // 2, :] + xpick
        #     # xdelnew = xall[:, num_depots + num_nodes // 2:num_depots + num_nodes, :] + xdel
        #     # emb_depot = xall[:, 0:num_depots, :]
        #     # x1 = torch.cat([emb_depot, xpicknew, xdelnew], 1)
        #     # x1 = x1.view(-1, self.hidden_node_dim)
        #     # x = x + x1
        #     # x = self.b5(x)
        #     # x = self.batch_norm((x + self.feedforward(x)))
        #
        #     # x = torch.cat([xall[:, :num_depots, :],
        #     #                xall[:, num_depots:num_depots + num_nodes // 2, :] + xpick,
        #     #                xall[:, num_depots + num_nodes // 2:num_depots + num_nodes, :] + xdel], 1).view(-1, self.hidden_node_dim)
        #     x = self.batch_norm(x + self.feedforward(x))

        return xd, xr


class FixedAttentionContext:
    def __init__(self, glimpse_key_d, glimpse_val_d, glimpse_key_r, glimpse_val_r):
        self.glimpse_key_d = glimpse_key_d
        self.glimpse_val_d = glimpse_val_d
        self.glimpse_key_r = glimpse_key_r
        self.glimpse_val_r = glimpse_val_r

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return FixedAttentionContext(
                self.glimpse_key_d[:, key],
                self.glimpse_val_d[:, key],
                self.glimpse_key_r[:, key],
                self.glimpse_val_r[:, key]
            )
        return super().__getitem__(key)


class Attention1(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        super(Attention1, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask, precomputed_k=None, precomputed_v=None):
        '''
        :param state_t: (batch_size, num_agents, input_dim * cat)
        :param context: (batch_size, n_nodes, input_dim)
        :param mask: selected nodes (batch_size, n_nodes)
        :param precomputed_k: Precomputed key tensor (optional)
        :param precomputed_v: Precomputed value tensor (optional)
        :return:
        '''
        batch_size, num_agents, _ = state_t.size()
        n_nodes = context.size(1)
        input_dim = self.input_dim

        # Compute Q, K, V
        Q = self.w(state_t).view(batch_size, num_agents, self.n_heads, self.head_dim)
        if precomputed_k is None or precomputed_v is None:
            # If no precomputed values, compute K and V
            K = self.k(context).view(batch_size, n_nodes, self.n_heads, self.head_dim)
            V = self.v(context).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        else:
            # Use precomputed values
            K = precomputed_k
            V = precomputed_v

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Compute compatibility
        compatibility = self.norm * torch.matmul(Q, K.transpose(2,
                                                                3))  # (batch_size, n_heads, num_agents, head_dim) * (batch_size, n_heads, head_dim, n_nodes)

        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float(-10000))

        scores = F.softmax(u_i, dim=-1)  # (batch_size, n_heads, num_agents, n_nodes)
        scores = scores.unsqueeze(3)

        out_put = torch.matmul(scores, V.unsqueeze(
            2))  # (batch_size, n_heads, num_agents, 1, n_nodes) * (batch_size, n_heads, 1, n_nodes, head_dim)
        out_put = out_put.squeeze(3).view(batch_size, num_agents,
                                          self.hidden_dim)  # (batch_size, num_agents, hidden_dim)

        out_put = self.fc(out_put)
        return out_put  # (batch_size, num_agents, hidden_dim)


class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = Attention1(n_heads, 1, input_dim, hidden_dim)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, context_d, n_d, mask, T, fixed_context=None):
        '''
        :param state_t: (batch_size, num_agents, input_dim * 3)
        :param context: (batch_size, n_nodes, input_dim)
        :param context_d: (batch_size, n_d, input_dim)
        :param mask: selected nodes (batch_size, num_agents, n_nodes)
        :param T: Temperature for softmax
        :param fixed_context: Precomputed attention keys and values (FixedAttentionContext)
        :return: softmax_score
        '''
        batch_size, num_agents, _ = state_t.size()
        n_nodes = context.size(1)

        if fixed_context is not None:
            # Use precomputed values
            precomputed_k_d = fixed_context.glimpse_key_d
            precomputed_v_d = fixed_context.glimpse_val_d
            precomputed_k_r = fixed_context.glimpse_key_r
            precomputed_v_r = fixed_context.glimpse_val_r
        else:
            precomputed_k_d, precomputed_v_d = None, None
            precomputed_k_r, precomputed_v_r = None, None

        # Compute attention for drones
        x_d = self.mhalayer(state_t[:, 0:n_d, :], context_d, mask[:, 0:n_d, :], precomputed_k=precomputed_k_d,
                            precomputed_v=precomputed_v_d)
        # Compute attention for robots
        x_r = self.mhalayer(state_t[:, n_d:num_agents, :], context, mask[:, n_d:num_agents, :],
                            precomputed_k=precomputed_k_r, precomputed_v=precomputed_v_r)

        Qd = x_d.view(batch_size, n_d, -1)
        Qr = x_r.view(batch_size, num_agents - n_d, -1)
        Kd = self.k(context_d).view(batch_size, context_d.size(1), -1)
        Kr = self.k(context).view(batch_size, context.size(1), -1)

        compatibilityd = self.norm * torch.matmul(Qd, Kd.transpose(1, 2))  # (batch_size, num_agents, n_nodes)
        compatibilityr = self.norm * torch.matmul(Qr, Kr.transpose(1, 2))  # (batch_size, num_agents, n_nodes)
        compatibility = torch.cat([compatibilityd, compatibilityr], 1)

        mask = mask.expand_as(compatibility)
        compatibility = compatibility.masked_fill(mask.bool(), float(-10000))

        scores = F.softmax(compatibility / T, dim=-1)  # (batch_size, num_agents, n_nodes)
        return scores


class Decoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.prob = ProbAttention(8, input_dim, hidden_dim)

        self.fcr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.fc1r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fcd = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.fc1d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wd = nn.Linear(hidden_dim + 3, hidden_dim,
                            bias=False)  # Linear layer that expects (hidden_dim + 3) input size
        self.Wr = nn.Linear(hidden_dim + 3, hidden_dim, bias=False)  # Linear layer for robots
        self.bd = nn.BatchNorm1d(hidden_dim)
        self.br = nn.BatchNorm1d(hidden_dim)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def _precompute(self, emb_d, emb_r):
        # Precompute the attention keys and values for drones and robots
        glimpse_key_d = self.prob.mhalayer.k(emb_d).view(emb_d.size(0), emb_d.size(1), self.prob.n_heads,
                                                         self.prob.hidden_dim // self.prob.n_heads)
        glimpse_val_d = self.prob.mhalayer.v(emb_d).view(emb_d.size(0), emb_d.size(1), self.prob.n_heads,
                                                         self.prob.hidden_dim // self.prob.n_heads)
        glimpse_key_r = self.prob.mhalayer.k(emb_r).view(emb_r.size(0), emb_r.size(1), self.prob.n_heads,
                                                         self.prob.hidden_dim // self.prob.n_heads)
        glimpse_val_r = self.prob.mhalayer.v(emb_r).view(emb_r.size(0), emb_r.size(1), self.prob.n_heads,
                                                         self.prob.hidden_dim // self.prob.n_heads)

        return FixedAttentionContext(glimpse_key_d, glimpse_val_d, glimpse_key_r, glimpse_val_r)

    def forward(self, emb_d, emb_r, pool_d, pool_r, capacity, demand, battery, time_window, num_depots, num_nodes,
                num_drones, num_robots, edge_attr_d, edge_attr_r, T, greedy=False):
        fixed_context = self._precompute(emb_d, emb_r)
        batch_size, _, hidden = emb_d.size()
        mask1 = emb_d.new_zeros((batch_size, num_drones + num_robots, emb_d.size(1)))
        mask = emb_d.new_zeros((batch_size, num_drones + num_robots, emb_d.size(1)))

        battery = battery.view(batch_size, num_drones + num_robots).unsqueeze(2).float().cuda()  # (batch_size, num_agents, 1)
        capacity = capacity.view(batch_size, num_drones + num_robots).unsqueeze(
            2).float()  # (batch_size, num_agents, 1)
        T_t = torch.zeros(batch_size, num_drones + num_robots).unsqueeze(2).float().cuda()  # (batch_size, num_agents, 1)
        demands = demand.view(batch_size, emb_d.size(1))
        time_window = time_window.view(batch_size, emb_d.size(1))
        edge_attr_d = edge_attr_d.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
        edge_attr_r = edge_attr_r.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
        # index = torch.zeros(batch_size, num_drones + num_robots).to(device).long()

        E_d = battery[0, 0].item()
        C_d = capacity[0, 0].item()
        E_r = battery[0, num_drones + num_robots - 1].item()
        C_r = capacity[0, num_drones + num_robots - 1].item()
        E = [E_d, E_r]
        C = [C_d, C_r]

        log_ps = []
        actions = []
        time_log = []
        BL = []

        i = 0
        while (mask1[:, :, num_depots:].max(dim=1)[0]).eq(0).any():

            if not (mask1[:, :, num_depots:].max(dim=1)[0]).eq(0).any() or i > num_nodes:
                break
            if i == 0:
                depot_indices = torch.randint(0, num_depots, (batch_size, num_drones + num_robots)).cuda()
                index = depot_indices

                s_t = torch.gather(emb_d, 1, depot_indices.unsqueeze(2).expand(-1, -1, emb_d.size(
                    2)))  # (batch_size, num_agents, hidden_dim)

                combined_input = torch.cat([s_t, capacity, T_t, battery], dim=-1)

                # Compute context vectors dynamically
                context_drones = self.bd(self.Wd(combined_input[:, :num_drones, :]).view(-1, hidden)).view(batch_size,
                                                                                                           num_drones,
                                                                                                           hidden)
                context_robots = self.br(self.Wr(combined_input[:, num_drones:, :]).view(-1, hidden)).view(batch_size,
                                                                                                           num_robots,
                                                                                                           hidden)
                context = torch.cat([context_drones, context_robots], dim=1)
                actions.append(index.data.unsqueeze(2))

            else:
                combined_input = torch.cat([s_t, capacity, T_t, battery], dim=-1)

                # Compute context vectors dynamically
                context_drones = self.bd(self.Wd(combined_input[:, :num_drones, :]).view(-1, hidden)).view(batch_size,
                                                                                                           num_drones,
                                                                                                           hidden)
                context_robots = self.br(self.Wr(combined_input[:, num_drones:, :]).view(-1, hidden)).view(batch_size,
                                                                                                           num_robots,
                                                                                                           hidden)
                context = torch.cat([context_drones, context_robots], dim=1)

            veh_d_mean = pool_d + (self.fcd(context[:, :num_drones, :])).mean(dim=1)  # Mean across drones
            veh_r_mean = pool_r + (self.fcr(context[:, num_drones:, :])).mean(dim=1)  # Mean across robots

            # Expand and add means
            decoder_input = torch.cat([
                veh_d_mean.unsqueeze(1).expand(batch_size, num_drones, hidden) + context[:, :num_drones, :],
                # Add mean for drones
                veh_r_mean.unsqueeze(1).expand(batch_size, num_robots, hidden) + context[:, num_drones:, :]
                # Add mean for robots
            ], dim=1)

            if i == 0:
                mask, mask1 = update_mask(demands, time_window, capacity, T_t, battery, num_drones, index, mask1, E, i)

            p = self.prob(decoder_input, emb_r, emb_d, num_drones, mask, T, fixed_context=fixed_context)
            p = torch.where(torch.isnan(p), torch.tensor(0.0001, dtype=p.dtype), p)
            dist = Categorical(p)
            if greedy:
                # val, index = p.max(dim=-1)
                ask = torch.zeros(batch_size, num_nodes+num_depots, dtype=torch.bool).cuda()

                # Initialize a tensor to hold the selected indices
                # selected_indices = torch.full((batch_size, num_drones+num_robots), -1, dtype=torch.long)

                # Iterate over each agent to select the max index
                for agent in range(num_drones+num_robots):
                    # Mask out already selected nodes
                    masked_p = p.clone()
                    masked_p[:, agent, :] = masked_p[:, agent, :].masked_fill(ask, float('-inf'))

                    # Find the index of the maximum value in the nodes dimension
                    _, max_indices = masked_p[:, agent, :].max(dim=-1)

                    # Store the selected indices
                    index[:, agent] = max_indices

                    # Update the mask to mark these nodes as selected
                    ask[torch.arange(batch_size), max_indices] = True


                # index = resolve_conflicts(p, index, num_depots)
            else:
                # mask2 = torch.where(mask1 == 1, mask1, mask)
                # non_depot_mask = mask2.clone()
                # non_depot_mask[:, :, :num_depots] = 0  # Set depots mask to 0

                # p = torch.where(non_depot_mask != 0, 0, p + 0.001)
                index = dist.sample()
                # index = torch.multinomial(p.view(batch_size * (num_drones + num_robots), num_depots + num_nodes),
                #                           1).view(batch_size, num_drones + num_robots)
                # index = resolve_conflicts_multinomial(p, index, non_depot_mask, num_depots)
                # index = index[:, torch.randperm(index.size(1))]

            actions.append(index.data.unsqueeze(2))

            # batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, num_drones + num_robots)
            # agent_indices = torch.arange(num_drones + num_robots).unsqueeze(0).expand(batch_size,
            #                                                                           num_drones + num_robots)
            log_p = dist.log_prob(index)
            # log_p = torch.log(p[batch_indices, agent_indices, index] + 1e-4)
            is_done = (mask1[:, :, num_depots:].max(dim=1)[0].sum(1).unsqueeze(1).expand(batch_size,
                                                                                         num_drones + num_robots) >= (
                               emb_d.size(1) - num_depots)).float()

            log_p = log_p * (1. - is_done)
            log_ps.append(log_p.unsqueeze(2))

            capacity, T_t, battery = update_state(demands, time_window, battery, T_t, capacity, index, E, C, num_drones,
                                                  actions, edge_attr_r, edge_attr_d, i)
            capacity, T_t, battery = capacity.unsqueeze(2), T_t.unsqueeze(2), battery.unsqueeze(2)
            time_log.append(T_t)
            BL.append(battery)
            mask, mask1 = update_mask(demands, time_window, capacity, T_t, battery, num_drones, index, mask1, E, i)

            s_t = torch.cat([
                torch.gather(emb_d, 1, index[:, :num_drones].unsqueeze(2).expand(-1, -1, emb_d.size(2))),
                torch.gather(emb_r, 1, index[:, num_drones:].unsqueeze(2).expand(-1, -1, emb_r.size(2)))
            ], dim=1)

            i += 1

        actions = torch.cat(actions, dim=2)

        return actions, torch.cat(log_ps, dim=2).sum(dim=2), torch.cat(time_log, dim=2), torch.cat(BL, dim=2)



class Model(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
        super(Model, self).__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.decoder = Decoder1(hidden_node_dim, hidden_node_dim)
        # for param in self.parameters():
        #     param.requires_grad = True

    def forward(self, datas, num_drones, num_robots, greedy=False, T=1, checkpoint_encoder=False, training=False):
        # Checkpoint the encoder forward pass if checkpointing is enabled and we are in training mode
        if checkpoint_encoder and training:
            x_d, x_r = checkpoint.checkpoint(self.encoder, datas, use_reentrant=True)
        else:
            x_d, x_r = self.encoder(datas)  # (batch,seq_len,hidden_node_dim)
        # print(2)
        pooledx = x_d.mean(dim=1)
        pooledr = x_r.mean(dim=1)
        batch_size = datas['x'].shape[0]
        num_depots = torch.sum(datas['demand'] == 0).item() // batch_size
        num_nodes = torch.sum(datas['demand'] != 0).item() // batch_size
        demand = datas['demand']
        time_window = datas['time_window']
        capacity = datas['capacity']
        battery = datas['battery']
        edge_attr_d = datas['edge_attr_d']
        edge_attr_r = datas['edge_attr_r']

        actions, log_p, time, BL = self.decoder(x_d, x_r, pooledx, pooledr, capacity, demand, battery, time_window,
                                                num_depots, num_nodes, num_drones, num_robots, edge_attr_d, edge_attr_r,
                                                T, greedy)
        return actions, log_p, time, BL
