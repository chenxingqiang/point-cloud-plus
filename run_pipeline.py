import argparse
import os
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from pointmodel import get_model, get_loss, MyDataset
from pointtransform import PointTransformerCls, arg
import numpy as np
from tqdm import tqdm

# 设置默认数据类型为 float32
torch.set_default_dtype(torch.float32)

# Define default paths
DEFAULT_POINTNET_MODEL = os.path.join("Model", "PointNet（B）", "0.787.pth")
DEFAULT_TRANSFORMER_MODEL = os.path.join("Model", "PointTransformer（B）", "0.819.pth")

def get_device():
    """Get the best available device: MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_default_model_path(model_type):
    if model_type == "PointNet":
        return DEFAULT_POINTNET_MODEL
    else:
        return DEFAULT_TRANSFORMER_MODEL

def load_model(model_path, device, weights_only=True):
    """Load model with proper device mapping"""
    try:
        if weights_only:
            return torch.load(model_path, weights_only=True, map_location=device)
        return torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Warning: Failed to load model with weights_only={weights_only}. Error: {str(e)}")
        if weights_only:
            # Retry without weights_only
            return load_model(model_path, device, weights_only=False)
        raise

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
        data = data.to(device).float()  # 确保数据是 float32
        target = target.to(device).float()  # 确保目标是 float32
        
        if isinstance(model, get_model):
            data = data.transpose(2, 1)
        
        optimizer.zero_grad()
        pred, trans_feat = model(data)
        loss = criterion(pred, target.view(-1, 1), trans_feat)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data = data.to(device).float()  # 确保数据是 float32
            target = target.to(device).float()  # 确保目标是 float32
            
            if isinstance(model, get_model):
                data = data.transpose(2, 1)
            
            pred, trans_feat = model(data)
            loss = criterion(pred, target.view(-1, 1), trans_feat)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def test(model_path, test_file, model_type="PointNet"):
    device = get_device()
    print(f"Using device: {device}")
    
    if model_type == "PointNet":
        model = get_model().to(device)
    else:
        cls = arg()
        model = PointTransformerCls(cls).to(device)
    
    # Add error handling for model loading
    if not os.path.exists(model_path):
        # Try with absolute path if relative path fails
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model file not found at {model_path} or {abs_path}")
        model_path = abs_path
    
    # Load model with proper device mapping
    try:
        state_dict = load_model(model_path, device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    model.eval()
    
    # Add error handling for test file
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found at {test_file}")
    
    # 使用 float32 而不是 float64
    point_data = np.loadtxt(test_file).astype(np.float32)
    point_data = torch.tensor(point_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    if model_type == "PointNet":
        point_data = point_data.transpose(2, 1)
    
    with torch.no_grad():
        pred, _ = model(point_data)
        return pred.cpu().numpy()[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], 
                      help='train or test mode')
    parser.add_argument('--model_type', type=str, default='PointNet',
                      choices=['PointNet', 'PointTransformer'], help='model type')
    parser.add_argument('--train_dir', type=str, help='training data directory')
    parser.add_argument('--val_dir', type=str, help='validation data directory')
    parser.add_argument('--test_file', type=str, help='test file path')
    parser.add_argument('--model_path', type=str, help='path to save/load model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], 
                      help='force device type (optional)')
    
    opt = parser.parse_args()
    
    # Set device based on argument or auto-detect
    if opt.device:
        if opt.device == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        elif opt.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device(opt.device)
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Use default model path if not provided
    if not opt.model_path:
        opt.model_path = get_default_model_path(opt.model_type)
    
    if opt.mode == 'train':
        # Validate directories exist
        if not os.path.exists(opt.train_dir):
            raise FileNotFoundError(f"Training directory not found at {opt.train_dir}")
        if not os.path.exists(opt.val_dir):
            raise FileNotFoundError(f"Validation directory not found at {opt.val_dir}")
        
        # Create Model directory if it doesn't exist
        os.makedirs(os.path.dirname(opt.model_path), exist_ok=True)
        
        # Initialize model
        if opt.model_type == "PointNet":
            model = get_model().to(device)
        else:
            cls = arg()
            model = PointTransformerCls(cls).to(device)
        
        # Create data loaders
        train_dataset = MyDataset(opt.train_dir, trans=True)
        val_dataset = MyDataset(opt.val_dir, trans=False)
        
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, 
                              shuffle=False, num_workers=4)
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        criterion = get_loss()
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(1, opt.epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
            val_loss = validate(model, val_loader, criterion, device)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), opt.model_path)
                print(f'Saved best model with validation loss: {val_loss:.6f}')
    
    else:  # Test mode
        if not opt.test_file:
            raise ValueError("Test file path must be provided for test mode")
            
        try:
            prediction = test(opt.model_path, opt.test_file, opt.model_type)
            print(f"Prediction: {prediction}")
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"An error occurred during testing: {str(e)}")

if __name__ == "__main__":
    main() 