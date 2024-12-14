import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np

class TeamPerformanceGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels=64):
        super().__init__()
        
        # GAT layers for node-level analysis
        self.gat1 = GATv2Conv(num_node_features, hidden_channels, edge_dim=num_edge_features)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=num_edge_features)
        
        # Output layers
        self.node_predictor = torch.nn.Linear(hidden_channels, 1)  # Individual performance
        self.graph_predictor = torch.nn.Linear(hidden_channels, 1)  # Team performance
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Node-level processing
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Individual performance prediction
        node_pred = self.node_predictor(x)
        
        # Team performance prediction (graph-level)
        graph_features = global_mean_pool(x, batch)
        graph_pred = self.graph_predictor(graph_features)
        
        return node_pred, graph_pred

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader

def prepare_data(nodes_file='data/neo4j_nodes.csv', edges_file='data/neo4j_edges.csv', val_split=0.2):
    # Load and prepare data as before
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    # Prepare features
    node_features = torch.tensor(nodes_df[['0', '1', '2']].values, dtype=torch.float)
    edge_index = torch.tensor([edges_df['0'].values, edges_df['1'].values], dtype=torch.long)
    edge_attr = torch.tensor(edges_df[['2', '3']].values, dtype=torch.float)
    
    # Split indices for training and validation
    num_nodes = node_features.size(0)
    indices = np.arange(num_nodes)
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    
    # Create train and validation masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    # Create data object with masks
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_features[:, 2].view(-1, 1),
        train_mask=train_mask,
        val_mask=val_mask
    )
    
    return data

def train_model(model, data, num_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        node_pred, graph_pred = model(data.x, data.edge_index, data.edge_attr, 
                                    torch.zeros(data.x.size(0), dtype=torch.long))
        
        train_loss = F.mse_loss(node_pred[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            node_pred, graph_pred = model(data.x, data.edge_index, data.edge_attr,
                                        torch.zeros(data.x.size(0), dtype=torch.long))
            val_loss = F.mse_loss(node_pred[data.val_mask], data.y[data.val_mask])
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
    
    return model


def main():
    # Prepare data
    data = prepare_data()
    
    # Initialize model
    model = TeamPerformanceGNN(
        num_node_features=data.x.size(1),
        num_edge_features=data.edge_attr.size(1)
    )
    
    # Train model
    trained_model = train_model(model, data)
    
    # Save model
    torch.save(trained_model.state_dict(), 'models/team_performance_gnn.pth')
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        node_pred, graph_pred = model(data.x, data.edge_index, data.edge_attr,
                                    torch.zeros(data.x.size(0), dtype=torch.long))
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'employee_id': range(len(node_pred)),
            'predicted_performance': node_pred.numpy().flatten(),
            'actual_performance': data.y.numpy().flatten()
        })
        predictions_df.to_csv('data/predictions.csv', index=False)

if __name__ == "__main__":
    main()
