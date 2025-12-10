#!/usr/bin/env python3
"""
PyTorch implementation of Multi-input Neural Network
Equivalent to the C++ implementation in main.cpp

This script creates a neural network with:
- Two inputs: input0 (1:1:1:2) and input1 (1:1:4:2)
- Shared LSTM layer for input1
- Shared FC layer for input0
- Concatenation and two output heads
- Target outputs: double and square of first input0 value
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, List


class MultiInputNet(nn.Module):
    """Multi-input neural network equivalent to the C++ implementation"""
    
    def __init__(self):
        super(MultiInputNet, self).__init__()
        
        # Shared layers (equivalent to shared_lstm and shared_fc in C++)
        self.shared_lstm = nn.LSTM(input_size=2, hidden_size=2, batch_first=True)
        self.shared_fc = nn.Linear(2, 2)
        
        # Output layers
        self.output_1 = nn.Linear(4, 1)  # After concatenation (2+2=4)
        self.output_2 = nn.Linear(4, 1)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, input0: torch.Tensor, input1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process input0 through shared FC
        fc_out = self.shared_fc(input0)  # [batch, 1, 2]
        fc_out = self.relu(fc_out)
        
        # Process input1 through shared LSTM
        # input1 shape: [batch, 4, 2] -> need to reshape for LSTM
        batch_size = input1.size(0)
        lstm_out, (hidden, cell) = self.shared_lstm(input1)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep [batch, 2]
        
        # Concatenate the outputs
        concat_out = torch.cat([fc_out.squeeze(1), lstm_out], dim=1)  # [batch, 4]
        
        # Two output heads
        out1 = self.output_1(concat_out)  # [batch, 1]
        out2 = self.output_2(concat_out)  # [batch, 1]
        
        return out1, out2


class MultiInputDataLoader:
    """Data loader equivalent to the C++ MultiDataLoader"""
    
    def __init__(self, data_size: int = 8):
        self.data_size = data_size
        self.current_idx = 0
        
    def generate_batch(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of training data"""
        # Generate random input values between -1 and 1
        input0 = torch.FloatTensor(batch_size, 1, 2).uniform_(-1.0, 1.0)
        input1 = torch.FloatTensor(batch_size, 4, 2).uniform_(-1.0, 1.0)
        
        # Calculate target outputs: double and square of first input0 value
        first_input_values = input0[:, 0, 0]  # First value of input0 for each sample
        label1 = 2.0 * first_input_values.unsqueeze(1)  # Double
        label2 = first_input_values.pow(2).unsqueeze(1)  # Square
        
        return input0, input1, label1, label2
    
    def next(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Get next batch (equivalent to C++ next function)"""
        input0, input1, label1, label2 = self.generate_batch(batch_size)
        
        self.current_idx += 1
        last = (self.current_idx >= self.data_size)
        if last:
            self.current_idx = 0
            
        return input0, input1, label1, label2, last


def train_model(model: nn.Module, dataloader: MultiInputDataLoader, epochs: int = 200, learning_rate: float = 0.01):
    """Train the model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training started...")
    model.train()
    
    for epoch in range(epochs):
        total_loss1 = 0.0
        total_loss2 = 0.0
        batch_count = 0
        
        while True:
            input0, input1, label1, label2, last = dataloader.next(batch_size=1)
            
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = model(input0, input1)
            
            # Calculate losses separately
            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)
            
            # Backward pass for each loss separately
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            batch_count += 1
            
            if last:
                break
        
        if (epoch + 1) % 10 == 0:
            avg_loss1 = total_loss1 / batch_count
            avg_loss2 = total_loss2 / batch_count
            avg_total_loss = avg_loss1 + avg_loss2
            print(f"Epoch {epoch+1}/{epochs} - Loss1: {avg_loss1:.6f}, Loss2: {avg_loss2:.6f}, Total: {avg_total_loss:.6f}")
    
    print("Training completed!")
    return model


def run_inference(model: nn.Module, dataloader: MultiInputDataLoader, num_samples: int = 4):
    """Run inference and print samples"""
    print("\n=== Running Inference and Printing Samples ===")
    print("Using trained model for inference")
    print(f"\nTesting {num_samples} samples:")
    print("-" * 40)
    
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            input0, input1, label1, label2, _ = dataloader.next(batch_size=1)
            
            # Forward pass
            output1, output2 = model(input0, input1)
            
            # Extract values for printing
            input0_vals = input0[0, 0].numpy()
            input1_vals = input1[0, 0].numpy()  # First timestep
            label1_val = label1[0, 0].item()
            label2_val = label2[0, 0].item()
            pred1_val = output1[0, 0].item()
            pred2_val = output2[0, 0].item()
            
            print(f"\nSample {i+1}:")
            print(f"Input0: [{input0_vals[0]:.6f}, {input0_vals[1]:.6f}]")
            print(f"Input1 (first timestep): [{input1_vals[0]:.6f}, {input1_vals[1]:.6f}]")
            print(f"Label1: {label1_val:.6f}, Predicted1: {pred1_val:.6f}")
            print(f"Label2: {label2_val:.6f}, Predicted2: {pred2_val:.6f}")
    
    print("\n" + "-" * 40)
    print("Inference completed!")


def main():
    """Main function"""
    print("PyTorch Multi-input Neural Network")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create model and data loader
    model = MultiInputNet()
    dataloader = MultiInputDataLoader(data_size=8)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    trained_model = train_model(model, dataloader, epochs=200, learning_rate=0.01)
    
    # Run inference
    run_inference(trained_model, dataloader, num_samples=4)


if __name__ == "__main__":
    main()
