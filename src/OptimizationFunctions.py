import torch
import os

def loss_function(input, output, target):
    penalty_lb = torch.sum(torch.where((input <= 0), (input)**2, torch.zeros_like(input)))
    penalty_ub = torch.sum(torch.where((input > 1), (input-1)**2, torch.zeros_like(input)))
    penalty = penalty_lb + penalty_ub
    real_loss = torch.nn.functional.mse_loss(output.real, target.real)
    return real_loss + penalty

def create_folders(path):
    # Check if the directory does not exist
    if not os.path.exists(path):
        # Create the directory
        # 'os.makedirs()' can create all the intermediate directories in the specified path
        os.makedirs(path)
        print(f'Directory "{path}" created')
    else:
        print(f'Directory "{path}" already exists')