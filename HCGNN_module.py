# Модуль по построению модели
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, HeteroConv
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle
import os
from PyIF import te_compute as te

class ChangePointDetector:
    def __init__(self, file_path, graph_path, graph_prefix, graph_type='correlation'):
        self.file_path = file_path
        self.graph_path = graph_path
        self.graph_prefix = graph_prefix
        self.graph_type = graph_type
        self.model = None
        self.optimizer = None
        self.num_epochs = 50
        self.hidden_channels = 16
        self.lr = 0.01
        self.window_size = 30
        self.split_ratio = 0.8

        self.data = self.load_real_data(self.file_path)
        self.train_data, self.test_data = self.split_data(self.data)
        self.normalized_train_data, self.mean, self.std = self.normalize_data(self.train_data)
        self.normalized_test_data = self.apply_normalization(self.test_data, self.mean, self.std)

        # Проверка наличия директории и графов, если нет, создание графов
        if not os.path.exists(self.graph_path) or not os.listdir(self.graph_path):
            os.makedirs(self.graph_path, exist_ok=True)
            if self.graph_type == 'correlation':
                self.create_and_save_graphs(self.normalized_train_data, 'train_')
                self.create_and_save_graphs(self.normalized_test_data, 'test_')
            elif self.graph_type == 'entropy':
                self.create_and_save_te_graphs(self.normalized_train_data, 'train_')
                self.create_and_save_te_graphs(self.normalized_test_data, 'test_')

        self.train_graphs = self.load_graphs(self.graph_path, 'train_' + self.graph_prefix)
        self.test_graphs = self.load_graphs(self.graph_path, 'test_' + self.graph_prefix)

        # Конвертируем графы в HeteroData
        self.train_hetero_graphs = self.convert_graphs_to_hetero_data(self.train_graphs)
        self.test_hetero_graphs = self.convert_graphs_to_hetero_data(self.test_graphs)

        if self.train_hetero_graphs:
            input_channels = self.train_hetero_graphs[0]['asset'].x.size(1)
            self.model = HeteroGNN(input_channels, self.hidden_channels)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            print("No training data available.")

    def load_real_data(self, file_path):
        df = pd.read_csv(file_path)
        data = df.drop(columns=['Date']).values.T
        self.asset_names = df.columns[1:].tolist()
        return data

    def normalize_data(self, data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        normalized_data = (data - mean) / std
        return normalized_data, mean, std

    def apply_normalization(self, data, mean, std):
        return (data - mean) / std

    def split_data(self, data):
        split_point = int(self.split_ratio * data.shape[1])
        return data[:, :split_point], data[:, split_point:]

    def create_and_save_graphs(self, data, prefix):
        num_nodes = data.shape[0]
        for t in range(self.window_size, data.shape[1]):
            window_data = data[:, t-self.window_size:t]
            corr_matrix = np.corrcoef(window_data)
            G = nx.Graph()
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if corr_matrix[i, j] > 0.5:
                        G.add_edge(i, j, weight=corr_matrix[i, j])
            for i in range(num_nodes):
                if not G.has_node(i):
                    G.add_node(i)
                G.nodes[i]['x'] = [data[i, t]]
            with open(os.path.join(self.graph_path, f'{prefix}{self.graph_prefix}_{t}.pkl'), 'wb') as f:
                pickle.dump(G, f)

    def create_and_save_te_graphs(self, data, prefix, threshold=0.01):
        num_nodes = data.shape[0]
        for t in tqdm(range(self.window_size, data.shape[1]), desc="Creating TE graphs"):
            window_data = data[:, t-self.window_size:t]
            G = nx.Graph()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        te_value = self.compute_transfer_entropy(window_data[i], window_data[j])
                        if te_value > threshold:
                            G.add_edge(i, j, weight=te_value)
            for i in range(num_nodes):
                if not G.has_node(i):
                    G.add_node(i)
                G.nodes[i]['x'] = [data[i, t]]
            with open(os.path.join(self.graph_path, f'{prefix}{self.graph_prefix}_{t}.pkl'), 'wb') as f:
                pickle.dump(G, f)

    def compute_transfer_entropy(self, source, target, k=1, embedding=1, safetyCheck=True, GPU=False):
        te_value = te.te_compute(source, target, k=k, embedding=embedding, safetyCheck=safetyCheck, GPU=GPU)
        return te_value

    def load_graphs(self, path, prefix):
        graphs = {}
        for file in os.listdir(path):
            if file.startswith(prefix) and file.endswith('.pkl'):
                date = file.split('_')[-1].replace('.pkl', '')
                with open(os.path.join(path, file), 'rb') as f:
                    graphs[date] = pickle.load(f)
        return graphs

    def convert_graphs_to_hetero_data(self, graphs):
        hetero_data_list = []
        for date, G in graphs.items():
            data = from_networkx(G, group_node_attrs=['x'])
            if 'x' not in data:
                print(f"No node features 'x' found in graph for date {date}")
                continue
            hetero_data = HeteroData()
            hetero_data['asset'].x = data['x'].float()
            hetero_data['asset', 'correlates_with', 'asset'].edge_index = data.edge_index
            hetero_data_list.append(hetero_data)
        return hetero_data_list

    def train_model(self):
        self.model.train()
        for epoch in tqdm(range(self.num_epochs), desc="Training Model"):
            epoch_loss = 0
            for graph in self.train_hetero_graphs:
                self.optimizer.zero_grad()
                out = self.model(graph)
                loss = F.mse_loss(out, graph['asset'].x)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_hetero_graphs)
            print(f'Epoch {epoch:03d}, Loss: {epoch_loss:.4f}')

    def detect_change_points(self):
        self.model.eval()
        residuals_list = []
        change_points = {i: [] for i in range(self.test_hetero_graphs[0]['asset'].x.size(0))}

        with torch.no_grad():
            for idx, graph in enumerate(self.test_hetero_graphs):
                node_embeddings = self.model(graph).detach().cpu().numpy()
                original_data = graph['asset'].x.cpu().numpy()

                residuals = np.abs(node_embeddings - original_data)
                residuals_mean = np.mean(residuals, axis=1)
                residuals_list.append(residuals_mean)

        residuals_mean_all = np.concatenate(residuals_list)
        global_median = np.median(residuals_mean_all)
        global_std = np.std(residuals_mean_all)

        for idx, residuals_mean in enumerate(residuals_list):
            for i in range(residuals_mean.shape[0]):
                if residuals_mean[i] > global_median + 3 * global_std:
                    change_points[i].append(idx)

        return change_points, np.array(residuals_list)

    def plot_individual_series(self, change_points, asset_name):
        asset_index = self.asset_names.index(asset_name)
        
        train_values = np.concatenate([graph['asset'].x.cpu().numpy()[asset_index] for graph in self.train_hetero_graphs])
        test_values = np.concatenate([graph['asset'].x.cpu().numpy()[asset_index] for graph in self.test_hetero_graphs])
        forecast_values_train = np.concatenate([self.model(graph).detach().cpu().numpy()[asset_index] for graph in self.train_hetero_graphs])
        forecast_values_test = np.concatenate([self.model(graph).detach().cpu().numpy()[asset_index] for graph in self.test_hetero_graphs])
        
        plt.figure(figsize=(15, 3))
        plt.plot(range(len(train_values)), train_values, label='Train Data')
        plt.plot(range(len(train_values), len(train_values) + len(test_values)), test_values, label='Test Data')
        plt.plot(range(len(forecast_values_train)), forecast_values_train, label='Train Forecast')
        plt.plot(range(len(train_values), len(train_values) + len(forecast_values_test)), forecast_values_test, label='Test Forecast')

        for cp in change_points[asset_index]:
            plt.axvline(x=len(train_values) + cp, color='r', linestyle='--')
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'{asset_name} - Change Point Detection')
        plt.legend()
        plt.show()

    def plot_residuals(self, residuals_list, change_points):
        plt.figure(figsize=(15, 3))
        plt.plot(np.arange(len(residuals_list)), np.mean(residuals_list, axis=1), label='Residuals')
        # for cp in change_points:
            # plt.axvline(x=cp, color='r', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Residuals')
        plt.title('Residuals Over Time')
        plt.legend()
        plt.show()

    def random_search(self, param_grid, n_iter=10):
        best_loss = float('inf')
        best_params = None

        for _ in range(n_iter):
            params = {key: random.choice(values) for key, values in param_grid.items()}
            self.hidden_channels = params['hidden_channels']
            self.lr = params['lr']
            self.num_epochs = params['num_epochs']

            self.train_model()
            change_points, residuals_list = self.detect_change_points()
            val_loss = self.evaluate_model()

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params

        return best_params, best_loss

    def evaluate_model(self):
        residuals_list = []
        self.model.eval()
        with torch.no_grad():
            for graph in self.train_hetero_graphs:
                node_embeddings = self.model(graph).detach().cpu().numpy()
                original_data = graph['asset'].x.detach().cpu().numpy()
                residuals = np.abs(node_embeddings - original_data)
                residuals_mean = np.mean(residuals, axis=1)
                residuals_list.append(residuals_mean)
        residuals_mean_all = np.concatenate(residuals_list)
        return np.mean(residuals_mean_all)

class HeteroGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(HeteroGNN, self).__init__()
        self.conv1 = HeteroConv({
            ('asset', 'correlates_with', 'asset'): GCNConv(input_channels, hidden_channels)
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('asset', 'correlates_with', 'asset'): GCNConv(hidden_channels, input_channels)
        }, aggr='sum')

    def forward(self, data):
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        return x_dict['asset']

