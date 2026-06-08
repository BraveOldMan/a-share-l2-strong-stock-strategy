import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from tqdm import tqdm

class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 32, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # A simple classification head for pretraining the embeddings
        self.classifier = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x shape: (Batch, Seq_len, Features)
        x = self.input_projection(x)
        # Pass through transformer
        out = self.transformer(x)
        # Pool the sequence (mean pooling)
        embedding = out.mean(dim=1)
        # Classification output
        logits = self.classifier(embedding)
        return logits, embedding

class TimeSeriesDataset(Dataset):
    def __init__(self, X_3d, y):
        self.X = torch.tensor(X_3d, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_3d_sequences(df: pl.DataFrame, features: list, seq_len: int = 5, max_samples: int = None):
    """
    Convert a flat 2D panel dataframe into 3D (N, seq_len, F) sequences by stock.
    Returns:
        X_3d: (N, seq_len, F) numpy array
        y: (N,) numpy array
        valid_indices: The row indices in the original dataframe that we kept (where seq_len was met)
    """
    # Sort chronologically by stock
    df = df.sort(["万得代码", "period_start"] if "period_start" in df.columns else ["万得代码", "自然日"])
    
    # Fill missing features with 0
    missing = [f for f in features if f not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(0.0).alias(f) for f in missing])
    
    X_raw = df.select(features).to_numpy()
    X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    
    # [V10] DL Robust Scaling: Prevent gradient explosion by standardizing features
    # Clip extreme values first before mean/std just in case of outliers
    X_raw = np.clip(X_raw, -1e8, 1e8) 
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0) + 1e-8
    X_raw = (X_raw - mean) / std
    X_raw = np.clip(X_raw, -10.0, 10.0)
    
    y_raw = df.select("label").to_numpy().flatten() if "label" in df.columns else np.zeros(len(df))
    y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
    stock_ids = df.select("万得代码").to_series().to_list()
    
    # We will build sequences step by step to avoid list append OOM
    valid_indices = []
    
    current_stock = None
    stock_idx_start = 0
    
    # Phase 1: Fast scan to collect valid target indices
    for i in range(len(df)):
        if stock_ids[i] != current_stock:
            current_stock = stock_ids[i]
            stock_idx_start = i
            
        if i - stock_idx_start + 1 >= seq_len:
            valid_indices.append(i)
            
    if max_samples and len(valid_indices) > max_samples:
        print(f"[Transformer] 序列总数 {len(valid_indices)} 过大，为了防止 OOM，在源头直接预采样至 {max_samples} 条...")
        valid_indices = np.random.choice(valid_indices, max_samples, replace=False).tolist()
        valid_indices.sort()
        
    num_valid = len(valid_indices)
    if num_valid == 0:
        return np.zeros((0, seq_len, len(features))), np.zeros(0), []
        
    # Phase 2: Pre-allocate target arrays to avoid massive peak RAM usage
    # This prevents the list from duplicating Memory size when copied to a Numpy Array
    X_3d = np.zeros((num_valid, seq_len, len(features)), dtype=np.float32)
    y_target = np.zeros(num_valid, dtype=np.float32)
    
    for row_idx, i in enumerate(valid_indices):
        X_3d[row_idx] = X_raw[i - seq_len + 1 : i + 1]
        y_target[row_idx] = y_raw[i]
        
    return X_3d, y_target, valid_indices

def train_transformer(df_train: pl.DataFrame, features: list, seq_len: int = 5, epochs: int = 10):
    """
    Train a temporal transformer on the dataset and return the model.
    """
    print(f"[Transformer] 构建 {seq_len} 天时序特征切片...")
    MAX_SEQ_LIMIT = 200000
    X_3d, y, _ = build_3d_sequences(df_train, features, seq_len=seq_len, max_samples=MAX_SEQ_LIMIT)
    
    if len(X_3d) == 0:
        print("[Transformer] 警告: 数据量不足以构建时序序列，跳过训练。")
        return None
        
    print(f"[Transformer] 最终训练集维度: {X_3d.shape}")
    
    dataset = TimeSeriesDataset(X_3d, y)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    model = TemporalFeatureExtractor(input_dim=len(features), embed_dim=32, num_heads=4, num_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # [V12] 排序标签为 0~10 的分位数，不能使用含 Logits 假设的 BCE (会计算越界爆 NaN)。改为对连续型均方误差拟合。
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(batch_x)
            loss = criterion(logits.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"    - Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | LR: {lr_now:.6f}")
        
    return model

def extract_transformer_embeddings(model, df: pl.DataFrame, features: list, seq_len: int = 5) -> np.ndarray:
    if model is None:
        return np.zeros((len(df), 32))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    X_3d, _, valid_indices = build_3d_sequences(df, features, seq_len=seq_len)
    
    out_embeddings = np.zeros((len(df), 32), dtype=np.float32)
    
    if len(X_3d) == 0:
        return out_embeddings
        
    # [V10 Patch] 取消在这里将整个 X_3d 一次性转为 Tensor，防止 RAM 一瞬间翻倍 OOM
    with torch.no_grad():
        chunk_size = 4096
        embs = []
        for i in range(0, len(X_3d), chunk_size):
            # 将 numpy slice 转为 tensor 并立刻给 GPU/CPU 消费，避开全局显存/内存峰值
            chunk_tensor = torch.tensor(X_3d[i:i+chunk_size], dtype=torch.float32).to(device)
            _, emb = model(chunk_tensor)
            embs.append(emb.cpu().numpy())
            
        all_embs = np.concatenate(embs, axis=0)
        
    # Scatter back to the original df positions
    out_embeddings[valid_indices] = all_embs
    return out_embeddings
