import pandas as pd
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import warnings
import itertools
import sys
from tqdm import tqdm

# Add explicit print statements with flushing
print("Script started", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"NumPy version: {np.__version__}", flush=True)
print(f"Pandas version: {pd.__version__}", flush=True)
print("Imports completed", flush=True)

warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("Starting script...", flush=True)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)}", flush=True)

# Define file path and directories
file_path = "DS Classification Case Data.csv"
TEMP_DIR = "temp_fraud_data"
MODELS_DIR = os.path.join(TEMP_DIR, "model_checkpoints")

print(f"Checking if data file exists: {os.path.exists(file_path)}", flush=True)
if os.path.exists(file_path):
    print(f"File size: {os.path.getsize(file_path) / (1024*1024*1024):.2f} GB", flush=True)
else:
    print(f"File not found: {file_path}", flush=True)
    sys.exit(1)

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Define location standardization mappings
COUNTRY_MAPPING = {
    'TURKEY': 'Turkey',
    'TÜRKİYE': 'Turkey',
    'TÜRKIYE': 'Turkey',
    'Türkiye': 'Turkey',
    'türkiye': 'Turkey',
    'turkey': 'Turkey',
    'TR': 'Turkey'
}

CITY_MAPPING = {
    'ISTANBUL': 'Istanbul',
    'İSTANBUL': 'Istanbul',
    'istanbul': 'Istanbul',
    'ANKARA': 'Ankara',
    'ankara': 'Ankara',
    'IZMIR': 'Izmir',
    'İZMIR': 'Izmir',
    'İZMİR': 'Izmir',
    'izmir': 'Izmir',
    'ANTALYA': 'Antalya',
    'antalya': 'Antalya'
}

# Model Architecture Definitions
class SimpleFraudNet(nn.Module):
    """Basic neural network for fraud detection"""
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super(SimpleFraudNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class DeepFraudNet(nn.Module):
    """Deeper neural network with batch normalization"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32]):
        super(DeepFraudNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class ResidualFraudNet(nn.Module):
    """Enhanced neural network with residual connections and attention mechanism"""
    def __init__(self, input_dim, hidden_dim=128, num_blocks=5, dropout=0.3):
        super(ResidualFraudNet, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm_input = nn.BatchNorm1d(hidden_dim)
        
        # Increase number of residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_blocks)
        ])
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Enhanced output layers with more capacity
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.batch_norm_input(x)
        x = torch.relu(x)
        
        for residual_block in self.residual_blocks:
            identity = x
            x = residual_block(x)
            x += identity
            x = torch.relu(x)
        
        # Apply attention
        attention_weights = torch.sigmoid(self.attention(x))
        x = x * attention_weights
        
        x = self.output_layer(x)
        return torch.sigmoid(x)

# Focal Loss implementation for imbalanced classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class FraudDataset(Dataset):
    """Dataset class for fraud detection"""
    def __init__(self, features, targets):
        print(f"Creating dataset with features shape: {features.shape}")
        
        # Make a copy to avoid modifying the original
        features_copy = features.copy()
        
        # Check for NaN values
        nan_count = features_copy.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in features. Replacing with 0.")
            features_copy = features_copy.fillna(0)
        
        # Convert all columns to float32 first
        for col in features_copy.columns:
            try:
                features_copy[col] = pd.to_numeric(features_copy[col], errors='coerce').astype('float32')
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {e}")
                features_copy[col] = 0  # Set to 0 if conversion fails
        
        # Now check for infinite values after conversion to numeric
        try:
            inf_count = np.sum(np.isinf(features_copy.values))
            if inf_count > 0:
                print(f"Warning: Found {inf_count} infinite values in features. Replacing with 0.")
                features_copy = features_copy.replace([np.inf, -np.inf], 0)
        except Exception as e:
            print(f"Warning: Could not check for infinite values: {e}")
            # Apply a safer method to replace potential infinities
            for col in features_copy.columns:
                max_val = 1e6
                features_copy[col] = features_copy[col].clip(-max_val, max_val)
        
        print(f"Processed features shape: {features_copy.shape}")
        
        # Final check for any remaining issues
        features_copy = features_copy.fillna(0)  # Ensure no NaNs remain
        
        self.features = torch.tensor(features_copy.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)
        
        print(f"Created tensor with shape: {self.features.shape}")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_and_evaluate_model(model, train_loader, val_loader, config, model_type):
    print(f"\nTraining {model_type} with config:", flush=True)
    print(f"Learning rate: {config['learning_rate']}", flush=True)
    print(f"Dropout: {config.get('dropout', 0.3)}", flush=True)
    
    if model_type == 'simple':
        print(f"Hidden dim 1: {config['hidden_dim1']}", flush=True)
        print(f"Hidden dim 2: {config['hidden_dim2']}", flush=True)
    elif model_type == 'deep':
        print(f"Hidden dims: {config['hidden_dims']}", flush=True)
    elif model_type == 'residual':
        print(f"Hidden dim: {config['hidden_dim']}", flush=True)
        print(f"Num blocks: {config.get('num_blocks', 5)}", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = model.to(device)
    
    # Use Focal Loss for better handling of imbalanced data
    alpha = 1.0 - (1.0 / config.get('pos_weight', 1.0))  # Calculate alpha from pos_weight
    alpha = max(0.05, min(0.95, alpha))  # Clamp alpha between 0.05 and 0.95
    
    print(f"Using Focal Loss with alpha={alpha}, gamma=2.0", flush=True)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 1e-4),
        betas=(0.9, 0.999)
    )
    
    # Use cosine annealing scheduler with warm restarts
    T_0 = 5  # Number of epochs per restart
    T_mult = 1  # Multiplier for T_0 after each restart
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=T_0, 
        T_mult=T_mult,
        eta_min=config['learning_rate'] / 100
    )
    
    best_val_auc = 0
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Check the first batch to debug dimensions
    print("Checking first batch dimensions...", flush=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"First batch data shape: {data.shape}", flush=True)
        print(f"First batch target shape: {target.shape}", flush=True)
        break
    
    # Use more epochs for better training
    max_epochs = 15  # Increased from 5 to 15
    print(f"Training for {max_epochs} epochs", flush=True)
    
    # Track metrics for each epoch
    train_losses = []
    val_aucs = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory allocated before training: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB", flush=True)
    
    for epoch in range(max_epochs):
        print(f"Starting epoch {epoch+1}/{max_epochs}", flush=True)
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Use tqdm for progress tracking
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}", flush=True)
                
            data, target = data.to(device), target.to(device)
            
            # Ensure input data is in valid range
            data = torch.clamp(data, min=-1e6, max=1e6)  # Prevent extreme values
            
            optimizer.zero_grad()
            
            # Debug print for first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"Model input shape: {data.shape}", flush=True)
                # Different model types have different attribute structures
                if model_type == 'simple' or model_type == 'deep':
                    print(f"Model expected input dim: {model.network[0].in_features}", flush=True)
                elif model_type == 'residual':
                    print(f"Model expected input dim: {model.input_layer.in_features}", flush=True)
            
            try:
                outputs = model(data)
                
                # Reshape outputs to match target shape
                outputs = outputs.squeeze()
                
                # Ensure target is float and binary
                target = target.float()
                target = torch.round(target).float()
                
                if epoch == 0 and batch_idx == 0:
                    print(f"Output shape after squeeze: {outputs.shape}", flush=True)
                    print(f"Target shape: {target.shape}", flush=True)
                
                loss = criterion(outputs, target)
                
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected at batch {batch_idx}. Skipping batch.", flush=True)
                    continue
                    
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}", flush=True)
                    
                # Periodically clear CUDA cache
                if torch.cuda.is_available() and batch_idx % 100 == 0 and batch_idx > 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_loss)
        
        # Update learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} completed with avg loss: {avg_loss:.4f}, lr: {current_lr:.6f}", flush=True)
        
        # Clear CUDA cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Validation phase
        print(f"Starting validation for epoch {epoch+1}", flush=True)
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")):
                if batch_idx % 50 == 0:
                    print(f"Validating batch {batch_idx}", flush=True)
                    
                data, target = data.to(device), target.to(device)
                data = torch.clamp(data, min=-1e6, max=1e6)  # Prevent extreme values
                
                try:
                    outputs = model(data)
                    outputs = outputs.squeeze()  # Reshape to match target
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"Computing validation metrics for epoch {epoch+1}", flush=True)
        try:
            # Check if there are multiple classes in the validation set
            unique_classes = len(set(val_targets))
            print(f"Number of unique classes in validation set: {unique_classes}", flush=True)
            
            if unique_classes < 2:
                print("Only one class in validation set, using accuracy instead of AUC", flush=True)
                # Use accuracy instead
                val_preds_binary = [1 if p > 0.5 else 0 for p in val_predictions]
                val_acc = sum(1 for p, t in zip(val_preds_binary, val_targets) if p == t) / len(val_targets)
                print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}", flush=True)
                val_auc = val_acc  # Use accuracy as a proxy for AUC
                val_aucs.append(val_auc)
            else:
                # Calculate ROC AUC
                val_auc = roc_auc_score(val_targets, val_predictions)
                val_aucs.append(val_auc)
                print(f"Epoch {epoch+1}, Validation AUC: {val_auc:.4f}", flush=True)
                
                # Calculate precision and recall at different thresholds
                precision, recall, thresholds = precision_recall_curve(val_targets, val_predictions)
                
                # Find threshold with best F1 score
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                best_threshold_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_threshold_idx] if len(thresholds) > best_threshold_idx else 0.5
                best_f1 = f1_scores[best_threshold_idx]
                best_precision = precision[best_threshold_idx]
                best_recall = recall[best_threshold_idx]
                
                val_precisions.append(best_precision)
                val_recalls.append(best_recall)
                val_f1s.append(best_f1)
                
                print(f"Best threshold: {best_threshold:.4f}", flush=True)
                print(f"At best threshold - Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}", flush=True)
                
        except Exception as e:
            print(f"Error computing validation metrics: {e}", flush=True)
            # Return a default value to continue training
            val_auc = 0.5
            val_aucs.append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(MODELS_DIR, f'{model_type}_checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'config': config,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}", flush=True)
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= 5:  # Increased patience
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break
    
    # Print summary of training
    print("\nTraining Summary:", flush=True)
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}", flush=True)
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(val_aucs) + 1), val_aucs)
        plt.title('Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        
        if len(val_f1s) > 0:
            plt.subplot(2, 2, 3)
            plt.plot(range(1, len(val_precisions) + 1), val_precisions, label='Precision')
            plt.plot(range(1, len(val_recalls) + 1), val_recalls, label='Recall')
            plt.title('Precision and Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(range(1, len(val_f1s) + 1), val_f1s)
            plt.title('F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'{model_type}_training_curves.png'))
        print(f"Saved training curves to {os.path.join(MODELS_DIR, f'{model_type}_training_curves.png')}", flush=True)
    except Exception as e:
        print(f"Could not plot training curves: {e}", flush=True)
    
    return best_val_auc

def grid_search_models(X_train, X_val, y_train, y_val, input_dim):
    """Perform grid search over model architectures and hyperparameters"""
    print(f"Input dimension: {input_dim}", flush=True)
    
    best_model = None
    best_config = None
    best_auc = 0.0
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train.astype(int))
    print(f"Class distribution in training data: {class_counts}", flush=True)
    
    # Calculate positive class weight for BCEWithLogitsLoss
    pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 1.0
    print(f"Positive class weight: {pos_weight}", flush=True)
    
    # Enhanced configurations for ResidualFraudNet only
    residual_configs = [
        {
            'learning_rate': 0.001,
            'dropout': 0.3,
            'hidden_dim': 128,
            'num_blocks': 5,
            'pos_weight': pos_weight,
            'weight_decay': 1e-4
        },
        {
            'learning_rate': 0.0005,
            'dropout': 0.4,
            'hidden_dim': 256,
            'num_blocks': 7,
            'pos_weight': pos_weight,
            'weight_decay': 1e-5
        },
        {
            'learning_rate': 0.0003,
            'dropout': 0.5,
            'hidden_dim': 192,
            'num_blocks': 6,
            'pos_weight': pos_weight,
            'weight_decay': 5e-5
        }
    ]
    
    # Create datasets for grid search
    print("Creating datasets for grid search...", flush=True)
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    
    # Calculate batch size based on available memory
    # Use smaller batch size for CUDA to avoid OOM errors
    batch_size = 32 if torch.cuda.is_available() else 64
    
    print(f"Creating data loaders with batch size {batch_size}...", flush=True)
    
    # Create weighted sampler for imbalanced data
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0,  # Avoid using multiple workers for debugging
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Avoid using multiple workers for debugging
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Only run ResidualFraudNet as it performs best
    print("\nTesting residual architecture...", flush=True)
    for config in residual_configs:
        print(f"Creating residual model with input_dim={input_dim}", flush=True)
        model = ResidualFraudNet(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_blocks=config.get('num_blocks', 5),
            dropout=config.get('dropout', 0.3)
        )
        
        print(f"Training model with config: {config}", flush=True)
        val_auc = train_and_evaluate_model(model, train_loader, val_loader, config, 'residual')
        
        if val_auc > best_auc:
            best_model = model
            best_config = config
            best_auc = val_auc
    
    return best_model, best_config, best_auc

def enrich_batch_with_features(df_batch, stats):
    """Enrich a batch with engineered features using pre-computed statistics"""
    print("Enriching batch with features...")
    
    df_batch = clean_location_data(df_batch)
    
    df_batch['payment_date'] = pd.to_datetime(df_batch['payment_date'])
    
    # Extract date features
    df_batch['payment_year'] = df_batch['payment_date'].dt.year
    df_batch['payment_month'] = df_batch['payment_date'].dt.month
    df_batch['payment_day'] = df_batch['payment_date'].dt.day
    df_batch['payment_hour'] = df_batch['payment_date'].dt.hour
    df_batch['payment_minute'] = df_batch['payment_date'].dt.minute
    df_batch['payment_dayofweek'] = df_batch['payment_date'].dt.dayofweek
    df_batch['is_weekend'] = df_batch['payment_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add time-based features
    df_batch['is_night'] = df_batch['payment_hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)
    df_batch['is_morning'] = df_batch['payment_hour'].apply(lambda x: 1 if (x > 6 and x <= 12) else 0)
    df_batch['is_afternoon'] = df_batch['payment_hour'].apply(lambda x: 1 if (x > 12 and x <= 18) else 0)
    df_batch['is_evening'] = df_batch['payment_hour'].apply(lambda x: 1 if (x > 18 and x < 22) else 0)
    
    # Add merchant features
    df_batch['merchant_transaction_count'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('transaction_count', 0)
    )
    
    df_batch['merchant_avg_price'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('avg_price', 0)
    )
    
    df_batch['merchant_price_std'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('price_std', 0)
    )
    
    df_batch['merchant_fraud_rate'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('fraud_rate', 0)
    )
    
    # Create price-related features
    df_batch['price_to_avg_ratio'] = df_batch['price'] / df_batch['merchant_avg_price'].replace(0, 1)
    df_batch['price_zscore'] = (df_batch['price'] - df_batch['merchant_avg_price']) / df_batch['merchant_price_std'].replace(0, 1)
    df_batch['price_log'] = np.log1p(df_batch['price'])
    
    # Add transaction velocity features (if buyer_id is available)
    if 'buyer_id' in df_batch.columns:
        # Group by buyer_id and sort by payment_date
        df_batch = df_batch.sort_values(['buyer_id', 'payment_date'])
        
        # Calculate time since last transaction for each buyer
        df_batch['prev_payment_date'] = df_batch.groupby('buyer_id')['payment_date'].shift(1)
        df_batch['time_since_last_transaction'] = (df_batch['payment_date'] - df_batch['prev_payment_date']).dt.total_seconds() / 3600  # in hours
        
        # Fill NaN values (first transaction for each buyer)
        df_batch['time_since_last_transaction'] = df_batch['time_since_last_transaction'].fillna(1000)  # Large value for first transactions
        
        # Create features for transaction velocity
        df_batch['transaction_velocity_24h'] = df_batch['time_since_last_transaction'].apply(lambda x: 1 if x <= 24 else 0)
        df_batch['transaction_velocity_6h'] = df_batch['time_since_last_transaction'].apply(lambda x: 1 if x <= 6 else 0)
        df_batch['transaction_velocity_1h'] = df_batch['time_since_last_transaction'].apply(lambda x: 1 if x <= 1 else 0)
        
        # Drop temporary columns
        df_batch.drop(['prev_payment_date'], axis=1, inplace=True)
    
    # Add card-related features (if card_id is available)
    if 'card_id' in df_batch.columns:
        # Calculate card usage frequency
        card_counts = df_batch['card_id'].value_counts().to_dict()
        df_batch['card_usage_count'] = df_batch['card_id'].map(card_counts)
        
        # Calculate card fraud rate
        card_fraud = df_batch.groupby('card_id')['is_fraud_transaction'].mean().to_dict()
        df_batch['card_fraud_rate'] = df_batch['card_id'].map(card_fraud).fillna(0)
    
    # Drop original datetime columns
    df_batch.drop(['payment_date'], axis=1, inplace=True)
    
    # Drop unnecessary columns
    drop_cols = ['payment_id']
    encrypted_cols = ['buyer_name', 'buyer_surname', 'buyer_email', 'buyer_gsm']
    drop_cols.extend([col for col in encrypted_cols if col in df_batch.columns])
    
    df_batch.drop([col for col in drop_cols if col in df_batch.columns], axis=1, inplace=True)
    
    return df_batch

def clean_location_data(df):
    """Clean and standardize location data (buyer_city, buyer_country)"""
    print("Cleaning location data...")
    
    # Standardize country names
    if 'buyer_country' in df.columns:
        df['buyer_country'] = df['buyer_country'].astype(str)
        df['buyer_country'] = df['buyer_country'].apply(
            lambda x: COUNTRY_MAPPING.get(x, x) if x in COUNTRY_MAPPING else x
        )
        df['buyer_country'] = df['buyer_country'].apply(
            lambda x: x.title() if x not in COUNTRY_MAPPING.values() else x
        )
    
    # Standardize city names
    if 'buyer_city' in df.columns:
        df['buyer_city'] = df['buyer_city'].astype(str)
        df['buyer_city'] = df['buyer_city'].apply(
            lambda x: CITY_MAPPING.get(x, x) if x in CITY_MAPPING else x
        )
        df['buyer_city'] = df['buyer_city'].apply(
            lambda x: x.title() if x not in CITY_MAPPING.values() else x
        )
    
    return df

def count_rows(file_path):
    """Count the number of rows in the CSV file without loading it entirely"""
    print("Counting rows in the dataset...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            row_count = -1
            for row_count, _ in enumerate(f):
                if row_count % 1000000 == 0 and row_count > 0:
                    print(f"Counted {row_count} rows so far...")
        return row_count + 1
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, trying with latin-1 encoding...")
        with open(file_path, 'r', encoding='latin-1') as f:
            row_count = -1
            for row_count, _ in enumerate(f):
                if row_count % 1000000 == 0 and row_count > 0:
                    print(f"Counted {row_count} rows so far...")
        return row_count + 1

def process_in_batches(file_path, batch_size=100000, max_rows=None):
    """Process the dataset in batches to calculate aggregated statistics"""
    print(f"Processing data in batches of {batch_size} rows...")
    
    # Initialize statistics
    stats = {
        'merchant_stats': {},
        'payment_type_stats': {},
        'payment_channel_stats': {},
        'card_stats': {},
        'total_rows': 0,
        'fraud_count': 0
    }
    
    # Calculate total rows to process
    total_rows = count_rows(file_path) - 1  # Subtract header row
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)
    
    print(f"Processing {total_rows} rows in total")
    
    # Process in batches
    for i in range(0, total_rows, batch_size):
        end_row = min(i + batch_size, total_rows)
        print(f"Processing batch {i//batch_size + 1}: rows {i} to {end_row}")
        
        try:
            df_batch = pd.read_csv(file_path, skiprows=range(1, i+1) if i > 0 else None, 
                                  nrows=min(batch_size, end_row-i), header=0 if i == 0 else 0,
                                  encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying with latin-1 encoding...")
            df_batch = pd.read_csv(file_path, skiprows=range(1, i+1) if i > 0 else None, 
                                  nrows=min(batch_size, end_row-i), header=0 if i == 0 else 0,
                                  encoding='latin-1')
        
        # Clean location data
        df_batch = clean_location_data(df_batch)
        
        # Update total statistics
        stats['total_rows'] += len(df_batch)
        stats['fraud_count'] += df_batch['is_fraud_transaction'].sum()
        
        # Calculate merchant statistics
        merchant_batch = df_batch.groupby('merchant_id').agg({
            'payment_id': 'count',
            'price': ['sum', 'mean', 'std', 'min', 'max'],
            'is_fraud_transaction': 'sum'
        })
        
        merchant_batch.columns = ['_'.join(col).strip() for col in merchant_batch.columns.values]
        
        for merchant_id, row in merchant_batch.iterrows():
            if merchant_id not in stats['merchant_stats']:
                stats['merchant_stats'][merchant_id] = {
                    'transaction_count': 0,
                    'price_sum': 0,
                    'price_values': [],
                    'fraud_count': 0
                }
            
            stats['merchant_stats'][merchant_id]['transaction_count'] += row['payment_id_count']
            stats['merchant_stats'][merchant_id]['price_sum'] += row['price_sum']
            stats['merchant_stats'][merchant_id]['price_values'].extend(df_batch[df_batch['merchant_id'] == merchant_id]['price'].tolist())
            stats['merchant_stats'][merchant_id]['fraud_count'] += row['is_fraud_transaction_sum']
    
    # Calculate derived statistics
    print("Calculating derived statistics...")
    
    # Merchant fraud rates
    for merchant_id, merchant_data in stats['merchant_stats'].items():
        merchant_data['fraud_rate'] = merchant_data['fraud_count'] / merchant_data['transaction_count'] if merchant_data['transaction_count'] > 0 else 0
        merchant_data['avg_price'] = merchant_data['price_sum'] / merchant_data['transaction_count'] if merchant_data['transaction_count'] > 0 else 0
        
        if merchant_data['price_values']:
            merchant_data['price_std'] = np.std(merchant_data['price_values']) if len(merchant_data['price_values']) > 1 else 0
            merchant_data['price_min'] = min(merchant_data['price_values'])
            merchant_data['price_max'] = max(merchant_data['price_values'])
        else:
            merchant_data['price_std'] = 0
            merchant_data['price_min'] = 0
            merchant_data['price_max'] = 0
        
        del merchant_data['price_values']
    
    stats['overall_fraud_rate'] = stats['fraud_count'] / stats['total_rows'] if stats['total_rows'] > 0 else 0
    
    print(f"Processed {stats['total_rows']} rows in total")
    print(f"Overall fraud rate: {stats['overall_fraud_rate'] * 100:.4f}%")
    
    with open(os.path.join(TEMP_DIR, 'fraud_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    return stats

def main():
    print("Optimized Deep Learning Fraud Detection with Hyperparameter Tuning", flush=True)
    print("=============================================================", flush=True)
    
    # Ask user for parameters
    max_rows = input("Enter maximum number of rows to process (leave blank for all): ")
    max_rows = int(max_rows) if max_rows.strip() else None
    
    sample_for_training = input("Enter number of rows to use for training (recommended: 500000, leave blank for all processed rows): ")
    sample_for_training = int(sample_for_training) if sample_for_training.strip() else None
    
    batch_size = input("Enter batch size (recommended: 100000, leave blank for default): ")
    batch_size = int(batch_size) if batch_size.strip() else 100000
    
    print(f"Using max_rows={max_rows}, sample_for_training={sample_for_training}, batch_size={batch_size}", flush=True)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = os.path.join(MODELS_DIR, 'processing_checkpoint.pkl')
    if os.path.exists(checkpoint_path):
        print(f"Found existing processing checkpoint at {checkpoint_path}", flush=True)
        print(f"Checkpoint file size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB", flush=True)
        print("Do you want to resume? (y/n): ", flush=True)
        resume = input().lower().strip()
        print(f"User input for resume: '{resume}'", flush=True)
        if resume == 'y':
            print("Resuming from checkpoint...", flush=True)
            try:
                with open(checkpoint_path, 'rb') as f:
                    print("Loading checkpoint file...", flush=True)
                    checkpoint = pickle.load(f)
                    print("Checkpoint loaded successfully", flush=True)
                    X_train = checkpoint.get('X_train')
                    X_val = checkpoint.get('X_val')
                    y_train = checkpoint.get('y_train')
                    y_val = checkpoint.get('y_val')
                    common_cols = checkpoint.get('common_cols')
                    stats = checkpoint.get('stats')
                    
                    print(f"Loaded checkpoint with training data shape: {X_train.shape}", flush=True)
                    print(f"Validation data shape: {X_val.shape}", flush=True)
                    
                    # Skip to model training
                    print("Skipping data processing and proceeding to model training...", flush=True)
                    
                    # Scale features
                    print("Scaling features...", flush=True)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Save scaler
                    with open(os.path.join(TEMP_DIR, 'scaler.pkl'), 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    # Convert to PyTorch datasets
                    print("Creating PyTorch datasets...", flush=True)
                    train_dataset = FraudDataset(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)
                    val_dataset = FraudDataset(pd.DataFrame(X_val_scaled, columns=X_val.columns), y_val)
                    
                    # Skip to grid search
                    print("Starting grid search...", flush=True)
                    input_dim = X_train.shape[1]
                    best_model, best_config, best_auc = grid_search_models(X_train, X_val, y_train, y_val, input_dim)
                    
                    print("\nBest model configuration:", flush=True)
                    print(best_config, flush=True)
                    print(f"Best validation AUC: {best_auc:.4f}", flush=True)
                    
                    # Save best model and configuration
                    print("Saving model and configuration...", flush=True)
                    torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))
                    with open(os.path.join(MODELS_DIR, 'best_config.pkl'), 'wb') as f:
                        pickle.dump(best_config, f)
                    
                    print("\nOptimized fraud detection model completed!", flush=True)
                    print(f"Model and features saved in {MODELS_DIR} directory", flush=True)
                    
                    return
            except Exception as e:
                print(f"Error loading checkpoint: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print("Continuing with fresh data processing...", flush=True)
    
    print("Loading and preprocessing data...", flush=True)
    
    # Process data in batches
    if os.path.exists(os.path.join(TEMP_DIR, 'fraud_stats.pkl')):
        print("Loading pre-computed statistics...", flush=True)
        with open(os.path.join(TEMP_DIR, 'fraud_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
    else:
        print("Computing statistics from scratch...", flush=True)
        stats = process_in_batches(file_path, batch_size, max_rows)
    
    # Prepare data for training
    print(f"Preparing data for training using up to {sample_for_training} rows...", flush=True)
    
    rows_to_process = min(sample_for_training, stats['total_rows']) if sample_for_training is not None else stats['total_rows']
    batches_needed = (rows_to_process + batch_size - 1) // batch_size
    
    print(f"Will process {rows_to_process} rows in {batches_needed} batches", flush=True)
    
    X_train_parts = []
    X_val_parts = []
    y_train_parts = []
    y_val_parts = []
    
    rows_processed = 0
    for i in range(batches_needed):
        current_batch_size = min(batch_size, rows_to_process - rows_processed)
        if current_batch_size <= 0:
            break
            
        print(f"Processing batch {i+1}/{batches_needed} for training...", flush=True)
        
        try:
            skip_rows = range(1, i * batch_size + 1) if i > 0 else None
            print(f"Reading CSV with skiprows={skip_rows}, nrows={current_batch_size}", flush=True)
            df_batch = pd.read_csv(file_path, skiprows=skip_rows, nrows=current_batch_size, 
                                  header=0 if i == 0 else 0, encoding='utf-8')
            print(f"Read batch with shape: {df_batch.shape}", flush=True)
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying with latin-1 encoding...", flush=True)
            skip_rows = range(1, i * batch_size + 1) if i > 0 else None
            df_batch = pd.read_csv(file_path, skiprows=skip_rows, nrows=current_batch_size, 
                                  header=0 if i == 0 else 0, encoding='latin-1')
            print(f"Read batch with shape: {df_batch.shape}", flush=True)
        
        # Enrich batch with features
        print("Enriching batch with features...", flush=True)
        df_batch = enrich_batch_with_features(df_batch, stats)
        print(f"Enriched batch shape: {df_batch.shape}", flush=True)
        
        # One-hot encode categorical variables
        categorical_cols = df_batch.select_dtypes(include=['object']).columns.tolist()
        print(f"One-hot encoding {len(categorical_cols)} categorical columns", flush=True)
        df_batch = pd.get_dummies(df_batch, columns=categorical_cols, drop_first=True)
        print(f"After one-hot encoding: {df_batch.shape}", flush=True)
        
        # Split features and target
        X_batch = df_batch.drop('is_fraud_transaction', axis=1)
        y_batch = df_batch['is_fraud_transaction']
        
        # Split into train and validation
        try:
            print(f"Attempting stratified split with y_batch distribution: {y_batch.value_counts()}", flush=True)
            X_train_batch, X_val_batch, y_train_batch, y_val_batch = train_test_split(
                X_batch, y_batch, test_size=0.2, random_state=42, stratify=y_batch
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}", flush=True)
            print("Falling back to regular split", flush=True)
            X_train_batch, X_val_batch, y_train_batch, y_val_batch = train_test_split(
                X_batch, y_batch, test_size=0.2, random_state=42
            )
        
        print(f"Train batch: {X_train_batch.shape}, Val batch: {X_val_batch.shape}", flush=True)
        
        X_train_parts.append(X_train_batch)
        X_val_parts.append(X_val_batch)
        y_train_parts.append(y_train_batch)
        y_val_parts.append(y_val_batch)
        
        rows_processed += current_batch_size
        
        # Save checkpoint after each batch
        if i % 5 == 0 and i > 0:
            print(f"Creating checkpoint after batch {i+1}...", flush=True)
            # Get common columns across all batches processed so far
            common_train_cols = set(X_train_parts[0].columns)
            for df in X_train_parts[1:]:
                common_train_cols = common_train_cols.intersection(df.columns)
            
            common_val_cols = set(X_val_parts[0].columns)
            for df in X_val_parts[1:]:
                common_val_cols = common_val_cols.intersection(df.columns)
            
            temp_common_cols = list(common_train_cols.intersection(common_val_cols))
            
            # Create temporary combined datasets for checkpoint
            temp_X_train = pd.concat([df[temp_common_cols] for df in X_train_parts], ignore_index=True)
            temp_X_val = pd.concat([df[temp_common_cols] for df in X_val_parts], ignore_index=True)
            temp_y_train = pd.concat(y_train_parts, ignore_index=True)
            temp_y_val = pd.concat(y_val_parts, ignore_index=True)
            
            checkpoint = {
                'X_train': temp_X_train,
                'X_val': temp_X_val,
                'y_train': temp_y_train,
                'y_val': temp_y_val,
                'common_cols': temp_common_cols,
                'stats': stats,
                'batches_processed': i+1
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"Checkpoint saved with {len(temp_common_cols)} common features", flush=True)
        
        # Break after processing one batch for testing
        if sample_for_training is not None and sample_for_training <= batch_size:
            print("Using only one batch for testing purposes", flush=True)
            break
        
        # Clear memory after each batch
        if i % 2 == 1:
            print("Clearing memory...", flush=True)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Get common columns across all batches
    common_train_cols = set(X_train_parts[0].columns)
    for df in X_train_parts[1:]:
        common_train_cols = common_train_cols.intersection(df.columns)
    
    common_val_cols = set(X_val_parts[0].columns)
    for df in X_val_parts[1:]:
        common_val_cols = common_val_cols.intersection(df.columns)
    
    common_cols = list(common_train_cols.intersection(common_val_cols))
    print(f"Using {len(common_cols)} common features across all batches", flush=True)
    
    # Combine using only common columns
    X_train = pd.concat([df[common_cols] for df in X_train_parts], ignore_index=True)
    X_val = pd.concat([df[common_cols] for df in X_val_parts], ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)
    y_val = pd.concat(y_val_parts, ignore_index=True)
    
    print(f"Training data shape: {X_train.shape}", flush=True)
    print(f"Validation data shape: {X_val.shape}", flush=True)
    
    # Save final checkpoint
    checkpoint = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'common_cols': common_cols,
        'stats': stats,
        'batches_processed': batches_needed
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print("Final data processing checkpoint saved", flush=True)
    
    # Scale features
    print("Scaling features...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    with open(os.path.join(TEMP_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Convert to PyTorch datasets
    print("Creating PyTorch datasets...", flush=True)
    train_dataset = FraudDataset(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)
    val_dataset = FraudDataset(pd.DataFrame(X_val_scaled, columns=X_val.columns), y_val)
    
    # Clear memory before grid search
    print("Clearing memory before grid search...", flush=True)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Grid search for best model and hyperparameters
    print("Starting grid search...", flush=True)
    input_dim = X_train.shape[1]
    best_model, best_config, best_auc = grid_search_models(X_train, X_val, y_train, y_val, input_dim)
    
    print("\nBest model configuration:", flush=True)
    print(best_config, flush=True)
    print(f"Best validation AUC: {best_auc:.4f}", flush=True)
    
    # Save best model and configuration
    print("Saving model and configuration...", flush=True)
    torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))
    with open(os.path.join(MODELS_DIR, 'best_config.pkl'), 'wb') as f:
        pickle.dump(best_config, f)
    
    print("\nOptimized fraud detection model completed!", flush=True)
    print(f"Model and features saved in {MODELS_DIR} directory", flush=True)

print("About to call main()", flush=True)
if __name__ == "__main__":
    print("Calling main()", flush=True)
    main()
    print("Main completed", flush=True) 