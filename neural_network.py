import argparse
import copy
import json
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, in_dimension, out_dimension, hidden_layers, neurons_per_hidden_layer):
        super(NeuralNetwork, self).__init__()
        layers = []
        current_dim = in_dimension

        for _ in range(hidden_layers):
            linear = nn.Linear(current_dim, neurons_per_hidden_layer)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_normalization_stats(data):
    return data.mean(dim=0), data.std(dim=0)


def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)


def load_and_prep_data(data_path, device, train_size=100000, quick_test=False):
    data = dict(np.load(data_path))
    tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    if quick_test:
        # Quick test uses a tiny subset of data
        tensors['training_features'] = tensors['training_features'][:100]
        tensors['training_labels'] = tensors['training_labels'][:100]
        tensors['evaluation_features'] = tensors['evaluation_features'][:100]
        tensors['evaluation_labels'] = tensors['evaluation_labels'][:100]
    else:
        # Subset training data based on --train-size argument
        tensors['training_features'] = tensors['training_features'][:train_size]
        tensors['training_labels'] = tensors['training_labels'][:train_size]

    # Compute normalization statistics ONLY on the training subset
    feat_mu, feat_std = get_normalization_stats(tensors['training_features'])
    label_mu, label_std = get_normalization_stats(tensors['training_labels'])

    norm_data = {}
    for split in ['training', 'evaluation', 'testing']:
        # The quick test doesn't necessarily need to process testing data, but we do it to avoid KeyErrors
        if quick_test and split == 'testing':
            continue
        
        norm_data[f'{split}_features'] = normalize(tensors[f'{split}_features'], feat_mu, feat_std)
        norm_data[f'{split}_labels'] = normalize(tensors[f'{split}_labels'], label_mu, label_std)

    return norm_data, label_mu, label_std, feat_mu, feat_std


def train_single_run(args, norm_data, label_mu, label_std, feat_mu, feat_std, run_prefix=""):
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1. FIX: Move the entire normalized dataset to the GPU upfront
    train_features = norm_data['training_features'].to(device)
    train_labels = norm_data['training_labels'].to(device)
    eval_features = norm_data['evaluation_features'].to(device)
    eval_labels = norm_data['evaluation_labels'].to(device)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    eval_dataset = TensorDataset(eval_features, eval_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = NeuralNetwork(14, 1, args.hidden_layers, args.neurons).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_eval_rmse = float('inf')
    best_epoch = -1
    best_model_state = None
    history = {'train_loss': [], 'eval_rmse': []}

    if args.device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            # Removed the .to(device) transfer lines since data is already on the GPU
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.view_as(outputs))
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_features.size(0)

        epoch_train_loss /= len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        eval_preds_list = []
        eval_labels_list = []

        with torch.no_grad():
            for batch_features, batch_labels in eval_loader:
                preds = model(batch_features)
                eval_preds_list.append(preds.cpu())
                # Ensure labels are also brought back to CPU for proper concatenation and math
                eval_labels_list.append(batch_labels.cpu())

            eval_preds_norm = torch.cat(eval_preds_list)
            eval_labels_norm = torch.cat(eval_labels_list)

            # Unnormalize
            eval_preds = eval_preds_norm * label_std + label_mu
            eval_labels = eval_labels_norm * label_std + label_mu

            # Calculate metrics
            eval_mse = nn.functional.mse_loss(eval_preds, eval_labels.view_as(eval_preds)).item()
            eval_rmse = np.sqrt(eval_mse)
            history['eval_rmse'].append(eval_rmse)

            if eval_rmse < best_eval_rmse:
                best_eval_rmse = eval_rmse
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if args.device == 'cuda':
        torch.cuda.synchronize()
    training_time = time.time() - start_time

    run_id = f"{run_prefix}hl{args.hidden_layers}_n{args.neurons}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)

    results = {
        "hidden_layers": args.hidden_layers,
        "neurons": args.neurons,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "best_eval_rmse": float(best_eval_rmse),
        "training_time": training_time
    }

    with open(os.path.join(args.out_dir, f"{run_id}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    np.savez(os.path.join(args.out_dir, f"{run_id}_history.npz"),
             train_loss=history['train_loss'], eval_rmse=history['eval_rmse'])

    torch.save(best_model_state, os.path.join(args.out_dir, f"{run_id}_model.pt"))

    return results, history['eval_rmse']


def evaluate_test_set(args, norm_data, label_mu, label_std):
    device = torch.device(args.device)
    
    # 2. FIX: Move testing data to GPU upfront as well
    test_features = norm_data['testing_features'].to(device)
    test_labels = norm_data['testing_labels'].to(device)
    
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    mse_list = []
    rmse_list = []

    print(f"\nEvaluating final configuration (HL: {args.hidden_layers}, Neurons: {args.neurons}) on held-out test set...")
    
    for seed in [0, 1, 2, 3, 4]:
        run_id = f"hl{args.hidden_layers}_n{args.neurons}_s{seed}"
        model_path = os.path.join(args.out_dir, f"{run_id}_model.pt")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Ensure training has completed for this seed.")
            continue

        model = NeuralNetwork(14, 1, args.hidden_layers, args.neurons).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_preds_list = []
        test_labels_list = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                preds = model(batch_features)
                test_preds_list.append(preds.cpu())
                test_labels_list.append(batch_labels.cpu())

        test_preds_norm = torch.cat(test_preds_list)
        test_labels_norm = torch.cat(test_labels_list)

        test_preds = test_preds_norm * label_std + label_mu
        test_labels = test_labels_norm * label_std + label_mu

        test_mse = nn.functional.mse_loss(test_preds, test_labels.view_as(test_preds)).item()
        test_rmse = np.sqrt(test_mse)
        
        mse_list.append(test_mse)
        rmse_list.append(test_rmse)

    if mse_list:
        print("\n=== Final Test Set Results ===")
        print(f"Mean Test MSE:  {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f} L^2")
        print(f"Mean Test RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f} L\n")


def generate_plots(all_histories, epochs, out_dir):
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    epoch_axis = np.arange(epochs)

    # Plot 1: Effect of Network Depth (Fix neurons=64, Vary hidden_layers)
    plt.figure(figsize=(10, 6))
    for hl in [1, 2, 3]:
        key = f"hl{hl}_n64"
        if key in all_histories and len(all_histories[key]) > 0:
            data = np.array(all_histories[key])
            mean_rmse = np.mean(data, axis=0)
            std_rmse = np.std(data, axis=0)

            plt.plot(epoch_axis, mean_rmse, label=f'{hl} Hidden Layers')
            plt.fill_between(epoch_axis, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2)

    plt.title('Effect of Network Depth on Evaluation RMSE (64 Neurons/Layer)')
    plt.xlabel('Epoch')
    plt.ylabel('Evaluation RMSE (Liters)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'plots', 'depth_effect_rmse.png'))
    plt.close()

    # Plot 2: Effect of Network Width (Fix hidden_layers=2, Vary neurons)
    plt.figure(figsize=(10, 6))
    for n in [32, 64, 128, 256]:
        key = f"hl2_n{n}"
        if key in all_histories and len(all_histories[key]) > 0:
            data = np.array(all_histories[key])
            mean_rmse = np.mean(data, axis=0)
            std_rmse = np.std(data, axis=0)

            plt.plot(epoch_axis, mean_rmse, label=f'{n} Neurons/Layer')
            plt.fill_between(epoch_axis, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2)

    plt.title('Effect of Network Width on Evaluation RMSE (2 Hidden Layers)')
    plt.xlabel('Epoch')
    plt.ylabel('Evaluation RMSE (Liters)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'plots', 'width_effect_rmse.png'))
    plt.close()


def run_gpu_sweep(args):
    device = torch.device(args.device)
    norm_data, label_mu, label_std, feat_mu, feat_std = load_and_prep_data(args.data_path, device, args.train_size)

    hidden_layers_list = [1, 2, 3]
    neurons_list = [32, 64, 128, 256]
    seeds = [0, 1, 2, 3, 4]

    all_results = []
    all_histories = {}
    total_runs = len(hidden_layers_list) * len(neurons_list) * len(seeds)
    current_run = 0

    print(f"Starting GPU Architecture Sweep: {total_runs} total runs.")

    for hl in hidden_layers_list:
        for n in neurons_list:
            config_key = f"hl{hl}_n{n}"
            all_histories[config_key] = []

            for seed in seeds:
                current_run += 1
                print(f"[{current_run}/{total_runs}] Running HL: {hl}, Neurons: {n}, Seed: {seed}...")

                args.hidden_layers = hl
                args.neurons = n
                args.seed = seed

                run_stats, eval_rmse_history = train_single_run(args, norm_data, label_mu, label_std, feat_mu, feat_std)
                all_results.append(run_stats)
                all_histories[config_key].append(eval_rmse_history)

    csv_path = os.path.join(args.out_dir, 'aggregated_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['hidden_layers', 'neurons', 'seed', 'batch_size', 'best_epoch', 'best_eval_rmse', 'training_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            writer.writerow(res)

    print(f"\nSweep complete. Aggregated results saved to {csv_path}")
    print("Generating plots summarizing the 5 replicates...")
    generate_plots(all_histories, args.epochs, args.out_dir)
    print(f"Plots saved to {args.out_dir}/plots/")


def run_self_directed_sweep(args):
    """
    Self-Directed Investigation: Tests the impact of batch size on convergence speed and performance.
    Baseline: Batch size 10000. New Condition: Batch size 1000.
    """
    device = torch.device(args.device)
    norm_data, label_mu, label_std, feat_mu, feat_std = load_and_prep_data(args.data_path, device, args.train_size)

    # Lock architecture to a baseline model
    args.hidden_layers = 2
    args.neurons = 64
    seeds = [0, 1, 2, 3, 4]
    batch_sizes = [10000, 1000]

    all_histories = {}
    print(f"Starting Self-Directed Investigation on Batch Size...")

    for bs in batch_sizes:
        args.batch_size = bs
        config_key = f"bs_{bs}"
        all_histories[config_key] = []
        
        for seed in seeds:
            args.seed = seed
            print(f"Running Self-Directed: Batch Size {bs}, Seed {seed}...")
            
            # Using a prefix to avoid overwriting baseline model checkpoint files
            _, eval_rmse_history = train_single_run(
                args, norm_data, label_mu, label_std, feat_mu, feat_std, run_prefix=f"sd_bs{bs}_"
            )
            all_histories[config_key].append(eval_rmse_history)

    # Plot self-directed results
    os.makedirs(os.path.join(args.out_dir, 'plots'), exist_ok=True)
    epoch_axis = np.arange(args.epochs)
    plt.figure(figsize=(10, 6))

    for bs in batch_sizes:
        key = f"bs_{bs}"
        if key in all_histories and len(all_histories[key]) > 0:
            data = np.array(all_histories[key])
            mean_rmse = np.mean(data, axis=0)
            std_rmse = np.std(data, axis=0)
            plt.plot(epoch_axis, mean_rmse, label=f'Batch Size {bs}')
            plt.fill_between(epoch_axis, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2)

    plt.title('Self-Directed Investigation: Effect of Batch Size on Evaluation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Evaluation RMSE (Liters)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, 'plots', 'self_directed_batch_size.png'))
    plt.close()
    
    print(f"Self-directed sweep complete. Plot saved to {args.out_dir}/plots/self_directed_batch_size.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swept Volume Neural Network Trainer")
    
    # Core variables
    parser.add_argument("--data-path", type=str, default="swept_volume_data.npz")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs")
    
    # Missing parameters implemented:
    parser.add_argument("--train-size", type=int, default=100000, help="Subset size of training data to use")
    parser.add_argument("--quick-test", action="store_true", help="Run a fast execution to verify code validity without testing set")
    parser.add_argument("--test-final", action="store_true", help="Evaluate a specific configuration's 5 seeds on the held-out testing set")
    
    # Execution modes
    parser.add_argument("--sweep", action="store_true", help="Run the full 60-configuration GPU sweep and plot results")
    parser.add_argument("--self-directed", action="store_true", help="Run the self-directed empirical investigation comparing batch sizes")

    # Config for individual runs
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # 1. Quick Test Execution
    if args.quick_test:
        print("Running in --quick-test mode...")
        args.epochs = 2 
        args.batch_size = 10 
        device = torch.device(args.device)
        norm_data, label_mu, label_std, feat_mu, feat_std = load_and_prep_data(args.data_path, device, args.train_size, quick_test=True)
        train_single_run(args, norm_data, label_mu, label_std, feat_mu, feat_std)
        print("Quick test complete. Exiting successfully.")
        exit(0)

    # 2. Final Test Set Evaluation
    elif args.test_final:
        device = torch.device(args.device)
        norm_data, label_mu, label_std, feat_mu, feat_std = load_and_prep_data(args.data_path, device, args.train_size)
        evaluate_test_set(args, norm_data, label_mu, label_std)

    # 3. Required GPU Sweep
    elif args.sweep:
        run_gpu_sweep(args)

    # 4. Self-Directed Investigation
    elif args.self_directed:
        run_self_directed_sweep(args)

    # 5. Default Individual Run
    else:
        device = torch.device(args.device)
        norm_data, label_mu, label_std, feat_mu, feat_std = load_and_prep_data(args.data_path, device, args.train_size)
        train_single_run(args, norm_data, label_mu, label_std, feat_mu, feat_std)
        print("Single run complete.")