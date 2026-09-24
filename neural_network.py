import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Class name and init parameters match required specifications.
class NeuralNetwork(nn.Module):
    def __init__(self, in_dimension, out_dimension, hidden_layers, neurons_per_hidden_layer):
        super(NeuralNetwork, self).__init__()
        layers = []
        current_dim = in_dimension

        for _ in range(hidden_layers):
            linear = nn.Linear(current_dim, neurons_per_hidden_layer)
            # Initialize linear-layer weights with xavier_uniform_ and biases with zeros_[cite: 1].
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            # torch.nn.ReLU() activation after every layer except the last[cite: 1].
            layers.append(nn.ReLU())
            current_dim = neurons_per_hidden_layer

        final_layer = nn.Linear(current_dim, out_dimension)
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    # Set the seed before constructing and initializing the network for reproducible replicates[cite: 1].
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_normalization_stats(data):
    return data.mean(dim=0), data.std(dim=0)


def normalize(data, mean, std):
    # Transform feature as (x - mu) / sigma[cite: 1].
    # Add a small epsilon to prevent division by zero in case of constant features
    return (data - mean) / (std + 1e-8)


def load_and_prep_data(data_path, train_size, device):
    data = dict(np.load(data_path))
    tensors = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}

    # Subset training data based on train_size parameter
    tensors['training_features'] = tensors['training_features'][:train_size]
    tensors['training_labels'] = tensors['training_labels'][:train_size]

    # Compute normalization statistics strictly from the training examples used[cite: 1].
    feat_mu, feat_std = get_normalization_stats(tensors['training_features'])
    label_mu, label_std = get_normalization_stats(tensors['training_labels'])

    # The evaluation and testing sets must be transformed using mean/std from training data[cite: 1].
    norm_data = {}
    for split in ['training', 'evaluation', 'testing']:
        norm_data[f'{split}_features'] = normalize(tensors[f'{split}_features'], feat_mu, feat_std)
        norm_data[f'{split}_labels'] = normalize(tensors[f'{split}_labels'], label_mu, label_std)

    return norm_data, label_mu, label_std, feat_mu, feat_std


def train_and_evaluate(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading data from {args.data_path}...")
    norm_data, label_mu, label_std, feat_mu, feat_std = load_and_prep_data(args.data_path, args.train_size, device)

    train_dataset = TensorDataset(norm_data['training_features'], norm_data['training_labels'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = NeuralNetwork(14, 1, args.hidden_layers, args.neurons).to(device)

    # Use torch.optim.Adam as the optimizer, with no weight decay[cite: 1].
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Use torch.nn.MSELoss() on the normalized labels as the training loss[cite: 1].
    criterion = nn.MSELoss()

    best_eval_rmse = float('inf')
    best_epoch = -1
    best_model_state = None
    history = {'train_loss': [], 'eval_rmse': []}

    if args.device == 'cuda':
        # call torch.cuda.synchronize() immediately before starting timer[cite: 1].
        torch.cuda.synchronize()
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_features.size(0)

        epoch_train_loss /= len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            eval_preds_norm = model(norm_data['evaluation_features'])
            # Convert predictions back to liters and report performance in the original units[cite: 1].
            eval_preds = eval_preds_norm * label_std + label_mu
            eval_labels = norm_data['evaluation_labels'] * label_std + label_mu

            eval_mse = nn.functional.mse_loss(eval_preds, eval_labels).item()
            eval_rmse = np.sqrt(eval_mse)
            history['eval_rmse'].append(eval_rmse)

            # Retain the model state from the epoch with the lowest evaluation error[cite: 1].
            if eval_rmse < best_eval_rmse:
                best_eval_rmse = eval_rmse
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch} | Train Loss (Norm): {epoch_train_loss:.4f} | Eval RMSE (Liters): {eval_rmse:.4f}")

    if args.device == 'cuda':
        # call torch.cuda.synchronize() immediately before stopping timer[cite: 1].
        torch.cuda.synchronize()
    training_time = time.time() - start_time

    print(
        f"\nTraining complete. Best Eval RMSE: {best_eval_rmse:.4f} at epoch {best_epoch}. Time: {training_time:.2f}s")

    # Save artifacts uniquely identifying the run configuration and seed[cite: 1].
    run_id = f"hl{args.hidden_layers}_n{args.neurons}_lr{args.lr}_bs{args.batch_size}_sz{args.train_size}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)

    results = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_eval_rmse": best_eval_rmse,
        "training_time": training_time,
        "normalization_stats": {
            "label_mu": label_mu.item(), "label_std": label_std.item(),
            "feat_mu": feat_mu.tolist(), "feat_std": feat_std.tolist()
        }
    }

    with open(os.path.join(args.out_dir, f"{run_id}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    np.savez(os.path.join(args.out_dir, f"{run_id}_history.npz"),
             train_loss=history['train_loss'], eval_rmse=history['eval_rmse'])

    torch.save(best_model_state, os.path.join(args.out_dir, f"{run_id}_model.pt"))
    print(f"Saved run artifacts to {args.out_dir}/")

    # Quick-test mode must not report performance on the held-out testing set[cite: 1].
    if not args.quick_test:
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            test_preds_norm = model(norm_data['testing_features'])
            test_preds = test_preds_norm * label_std + label_mu
            test_labels = norm_data['testing_labels'] * label_std + label_mu
            test_mse = nn.functional.mse_loss(test_preds, test_labels).item()
            test_rmse = np.sqrt(test_mse)
            print(f"Final Held-Out Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swept Volume Neural Network Trainer")
    # Command-line arguments specifying training-data size, learning rate, architecture, batch size, replicate seed, epochs, and device[cite: 1].
    parser.add_argument("--data-path", type=str, default="swept_volume_data.npz")
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run short training on small subset to verify pipeline")

    args = parser.parse_args()

    if args.quick_test:
        print("Running in quick-test mode...")
        args.train_size = 1000
        args.epochs = 5
        args.batch_size = 100

    train_and_evaluate(args)