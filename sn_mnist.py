"""
sn_mnist.py - MNIST classification using Sprecher Networks
"""

import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

# Import Sprecher Network components
from sn_core import SprecherMultiLayerNetwork
from sn_core.model import TRAIN_PHI_RANGE, USE_RESIDUAL_WEIGHTS

# Configuration
MODEL_FILE = 'sn_mnist_model.pth'
DATA_DIRECTORY = './data'

# Sprecher Network hyperparameters (easy to modify for experimentation)
SN_CONFIG = {
    'architecture': [100],  # Hidden layers between 784 inputs and 10 outputs
    'phi_knots': 50,        # Knots for monotonic splines
    'Phi_knots': 50,        # Knots for general splines
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
    'batch_size': 64,
    'epochs': 10,
}


def count_parameters_detailed(model):
    """Count parameters with detailed breakdown matching SN code style."""
    # Count different parameter types
    total_spline_knots = 0
    lambda_params = 0
    eta_params = 0
    residual_params = 0
    output_params = 2  # output_scale and output_bias
    phi_range_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'coeffs' in name or 'log_increments' in name:
            # Spline knots
            total_spline_knots += param.numel()
        elif 'lambdas' in name:
            # Lambda weight matrices
            lambda_params += param.numel()
        elif 'eta' in name:
            # Eta shift parameters
            eta_params += param.numel()
        elif 'residual_weight' in name or 'residual_projection' in name:
            # Residual connection parameters
            residual_params += param.numel()
        elif 'output_scale' in name or 'output_bias' in name:
            # Already counted above
            pass
        elif 'phi_range_params' in name:
            # Global range parameters (dc, dr, cc, cr)
            phi_range_params += param.numel()
    
    # Core parameters (excluding residual and output params)
    core_params = total_spline_knots + lambda_params + eta_params
    
    # Total parameters
    if TRAIN_PHI_RANGE:
        total_params = core_params + residual_params + output_params + phi_range_params
    else:
        total_params = core_params + residual_params + output_params
    
    # Count blocks
    num_blocks = len(model.sprecher_net.layers)
    
    return {
        'total': total_params,
        'core': core_params,
        'spline_knots': total_spline_knots,
        'lambda': lambda_params,
        'eta': eta_params,
        'residual': residual_params,
        'output': output_params,
        'phi_range': phi_range_params,
        'num_blocks': num_blocks
    }


def process_image(file_path):
    """Process an image file for inference."""
    image = Image.open(file_path)
    image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    data = np.asarray(image)
    
    # Check if the image is likely to have a white background
    if data.sum() > 127.5 * 28 * 28:
        data = 255 - data  # Invert the pixel values
    
    # Normalize to [0, 1] range for Sprecher Networks
    normalized_data = data / 255.0
    return normalized_data.flatten()


class MNISTSprecherNet(nn.Module):
    """Wrapper around SprecherMultiLayerNetwork for MNIST classification."""
    
    def __init__(self, architecture, phi_knots=100, Phi_knots=100):
        super().__init__()
        self.sprecher_net = SprecherMultiLayerNetwork(
            input_dim=784,  # 28x28 flattened
            architecture=architecture,
            final_dim=10,   # 10 digit classes
            phi_knots=phi_knots,
            Phi_knots=Phi_knots
        )
    
    def forward(self, x):
        # Flatten the input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.sprecher_net(x)


def train_epoch(model, train_loader, optimizer, loss_function, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            
            # Flatten and normalize to [0, 1]
            images = images.view(images.size(0), -1)
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            
            # Gradient clipping (as in SN code)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            tepoch.set_postfix(loss=epoch_loss/len(train_loader), acc=f"{accuracy:.1f}%")
    
    return epoch_loss / len(train_loader), accuracy


def test_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            # Flatten and normalize to [0, 1]
            images = images.view(images.size(0), -1)
            images = (images + 1) / 2
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def print_architecture_details(architecture, phi_knots, Phi_knots):
    """Print detailed architecture information."""
    print(f"\nSprecher Network Configuration:")
    print(f"  Architecture: 784 → {architecture} → 10")
    print(f"  φ knots per spline: {phi_knots}")
    print(f"  Φ knots per spline: {Phi_knots}")
    
    # Calculate number of blocks
    if len(architecture) == 0:
        num_blocks = 1
    else:
        num_blocks = len(architecture) + 1
    
    print(f"  Number of blocks: {num_blocks}")
    print(f"  Total φ splines: {num_blocks} (one per block)")
    print(f"  Total Φ splines: {num_blocks} (one per block)")
    print(f"  Total spline knots: {num_blocks * (phi_knots + Phi_knots)}")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get program mode
    while True:
        try:
            mode = int(input("Enter mode (0: train, 1: test, 2: infer): "))
            if mode in [0, 1, 2]:
                break
            else:
                print("Please enter 0, 1, or 2.")
        except ValueError:
            print("Please enter a valid integer.")
    
    if mode == 0:  # Training
        # Initialize model
        model = MNISTSprecherNet(
            architecture=SN_CONFIG['architecture'],
            phi_knots=SN_CONFIG['phi_knots'],
            Phi_knots=SN_CONFIG['Phi_knots']
        ).to(device)
        
        # Print architecture details
        print_architecture_details(
            SN_CONFIG['architecture'], 
            SN_CONFIG['phi_knots'], 
            SN_CONFIG['Phi_knots']
        )
        
        # Count and display parameters
        param_counts = count_parameters_detailed(model)
        print(f"\nTotal number of trainable parameters: {param_counts['total']:,}")
        print(f"  - Spline knots: {param_counts['spline_knots']:,}")
        print(f"  - Lambda (λ) weight matrices: {param_counts['lambda']:,}")
        print(f"  - Eta (η) shift parameters: {param_counts['eta']:,}")
        if USE_RESIDUAL_WEIGHTS and param_counts['residual'] > 0:
            print(f"  - Residual connection weights: {param_counts['residual']:,}")
        print(f"  - Output scale and bias: {param_counts['output']:,}")
        if TRAIN_PHI_RANGE and param_counts['phi_range'] > 0:
            print(f"  - Global range parameters (dc, dr, cc, cr): {param_counts['phi_range']:,}")
        
        # Check if model exists
        model_exists = os.path.exists(MODEL_FILE)
        retrain_from_scratch = False
        
        if model_exists:
            while True:
                try:
                    user_input = input("\nRetrain from scratch? (1: Yes, 0: No): ")
                    retrain_from_scratch = int(user_input.strip())
                    if retrain_from_scratch in [0, 1]:
                        retrain_from_scratch = bool(retrain_from_scratch)
                        break
                    else:
                        print("Please enter 0 or 1.")
                except ValueError:
                    print("Please enter a valid integer.")
            
            if retrain_from_scratch:
                os.remove(MODEL_FILE)
                print(f"Deleted existing model file: {MODEL_FILE}")
                if os.path.exists(DATA_DIRECTORY):
                    shutil.rmtree(DATA_DIRECTORY)
                    print(f"Deleted existing data directory: {DATA_DIRECTORY}")
        
        # Load model if continuing training
        if model_exists and not retrain_from_scratch:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device), strict=False)
            print("Loaded saved model for further training.")
        
        # Setup data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # This gives [-1, 1], we'll convert to [0, 1] later
        ])
        train_dataset = MNIST(root=DATA_DIRECTORY, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=SN_CONFIG['batch_size'], shuffle=True)
        
        # Setup optimizer (using AdamW like in SN code)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=SN_CONFIG['learning_rate'],
            weight_decay=SN_CONFIG['weight_decay']
        )
        loss_function = nn.CrossEntropyLoss()
        
        # Training loop
        print(f"\nStarting training for {SN_CONFIG['epochs']} epochs...")
        best_accuracy = 0
        
        for epoch in range(SN_CONFIG['epochs']):
            print(f"\nEpoch {epoch + 1}/{SN_CONFIG['epochs']}")
            avg_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_function, device)
            print(f"Average loss: {avg_loss:.4f}, Training accuracy: {train_acc:.2f}%")
            
            # Save model after each epoch
            torch.save(model.state_dict(), MODEL_FILE)
            
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                torch.save(model.state_dict(), MODEL_FILE.replace('.pth', '_best.pth'))
        
        print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
        
    elif mode == 1:  # Testing
        # Load model
        model = MNISTSprecherNet(
            architecture=SN_CONFIG['architecture'],
            phi_knots=SN_CONFIG['phi_knots'],
            Phi_knots=SN_CONFIG['Phi_knots']
        ).to(device)
        
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file {MODEL_FILE} not found. Please train first.")
            return
        
        # Load with strict=False to handle missing buffers
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device), strict=False)
        model.eval()
        
        # Print architecture and parameter count
        print_architecture_details(
            SN_CONFIG['architecture'], 
            SN_CONFIG['phi_knots'], 
            SN_CONFIG['Phi_knots']
        )
        
        param_counts = count_parameters_detailed(model)
        print(f"\nModel has {param_counts['total']:,} parameters")
        
        # Setup test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = MNIST(root=DATA_DIRECTORY, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=SN_CONFIG['batch_size'], shuffle=False)
        
        # Test
        accuracy = test_model(model, test_loader, device)
        print(f"\nTest accuracy: {accuracy:.2f}%")
        
    elif mode == 2:  # Inference
        # Load model
        model = MNISTSprecherNet(
            architecture=SN_CONFIG['architecture'],
            phi_knots=SN_CONFIG['phi_knots'],
            Phi_knots=SN_CONFIG['Phi_knots']
        ).to(device)
        
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file {MODEL_FILE} not found. Please train first.")
            return
        
        # Load with strict=False to handle missing buffers
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device), strict=False)
        model.eval()
        
        # Process image
        if not os.path.exists('digit.png'):
            print("Error: 'digit.png' not found. Please provide an image file.")
            return
        
        image_data = process_image('digit.png')
        image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            probs_list = probabilities.cpu().numpy().flatten()
            
            print("\nProbabilities:")
            for i, prob in enumerate(probs_list):
                print(f"  Digit {i}: {prob:.6f}")
            
            predicted = torch.argmax(probabilities, dim=1).item()
            confidence = probs_list[predicted] * 100
            print(f"\nPredicted digit: {predicted} (confidence: {confidence:.1f}%)")


if __name__ == "__main__":
    main()