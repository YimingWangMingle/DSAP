import torch
import torch.nn as nn
import torch.nn.functional as F

class StateAbstractionNetwork(nn.Module):
    def __init__(self, state_dim, feature_dim, position_dim):
        super(StateAbstractionNetwork, self).__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.position_dim = position_dim
        
        # Learnable position embedding
        self.position_embedding = nn.Parameter(torch.randn(1, state_dim, position_dim))
        
        # Encoder network
        self.encoder = nn.Linear(1 + position_dim, feature_dim)
        
        # Decoder network
        self.decoder = nn.Linear(feature_dim + position_dim, 1)

    def forward(self, state, adjacency_matrix):
        # Add position embedding to the input
        state_with_pos = torch.cat((state.unsqueeze(2), self.position_embedding.repeat(state.size(0), 1, 1)), dim=2)
        
        # Apply the shared encoder to each dimension
        features = F.relu(self.encoder(state_with_pos))
        
        # Multiply the features with the provided causal graph adjacency matrix
        filtered_features = torch.matmul(features, adjacency_matrix)
        
        # Pass through the decoder
        decoder_input = torch.cat((filtered_features, self.position_embedding.repeat(state.size(0), 1, 1)), dim=2)
        abstracted_state = self.decoder(decoder_input).squeeze(2)
        
        return abstracted_state
        
    def monte_carlo_abstraction(self, state, adjacency_matrices, K):
        # Ensure that the adjacency_matrices tensor has the correct shape
        assert adjacency_matrices.shape[0] == K and adjacency_matrices.shape[1] == self.feature_dim
        
        # Repeat the state across K samples for batch processing
        state_repeated = state.repeat(K, 1)
        
        # Perform Monte Carlo sampling
        abstracted_states = []
        for k in range(K):
            adjacency_matrix_k = adjacency_matrices[k]  # Select the k-th adjacency matrix
            abstracted_state_k = self.forward(state_repeated[k].unsqueeze(0), adjacency_matrix_k)
            abstracted_states.append(abstracted_state_k)
        
        # Stack all abstracted states and compute the mean
        abstracted_states_stack = torch.stack(abstracted_states)
        abstracted_state_mean = torch.mean(abstracted_states_stack, dim=0)
        
        return abstracted_state_mean.squeeze(0)  # Remove the extra dimension



# Hyperparameters
state_dim = 10  # Size of the state vector
feature_dim = 5  # Size of the feature vector after encoding
position_dim = 3  # Dimension of the position embedding
K = 100  # Number of Monte Carlo samples

# Create the state abstraction network
model = StateAbstractionNetwork(state_dim, feature_dim, position_dim)

# Example state tensor and a batch of adjacency matrices for Monte Carlo sampling
batch_size = 1
state = torch.randn(batch_size, state_dim)  # Random state tensor for demonstration
adjacency_matrices = torch.randn(K, feature_dim, feature_dim)  # Batch of adjacency matrices

# Perform Monte Carlo state abstraction
abstracted_state = model.monte_carlo_abstraction(state, adjacency_matrices, K)
print(abstracted_state)