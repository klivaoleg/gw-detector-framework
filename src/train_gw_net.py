#!/usr/bin/env python3
"""
Обучение 1D-CNN для классификации ГВ (Финальная рабочая версия)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

plt.style.use('dark_background')

class GWDataset(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        if not self.files:
            raise ValueError(f"CSV файлы не найдены в {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = os.path.join(self.folder, self.files[idx])
        df = pd.read_csv(filepath)
        
        data = df[['Det1_Strain', 'Det2_Strain']].values.T.astype(np.float64)
        label = float(df['Label'].iloc[0])

        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std = np.maximum(std, 1e-30)  # Защита от деления на ноль
        
        normalized_data = (data - mean) / std
        return torch.tensor(normalized_data, dtype=torch.float32), torch.tensor(label)

class GWClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2), 
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Обучение", leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        correct += ((outputs > 0).float() == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    return total_loss / total, correct / total

def validate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            correct += ((outputs > 0).float() == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / total, acc, auc

def main():
    print("🚀 Запуск обучения (Финальная версия)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Устройство: {device}")
    
    try:
        train_dataset = GWDataset('dataset/train')
        val_dataset = GWDataset('dataset/test')
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    print(f"📂 Train: {len(train_dataset)} | Test: {len(val_dataset)}")

    # Быстрая проверка нормализации
    sample_data, _ = next(iter(train_loader))
    print(f"📊 Min/Max входных данных: {sample_data.min().item():.3f} / {sample_data.max().item():.3f}")
    if sample_data.abs().max().item() < 0.5:
        print("⚠️ Данные слишком маленькие. Проверь генератор.")
    else:
        print("✅ Данные готовы. Запуск обучения...")

    model = GWClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_acc = 0.0
    epochs = 5
    history = {'t_loss': [], 'v_loss': [], 'v_acc': [], 'v_auc': []}
    
    print("\n" + "="*50)
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_auc = validate(model, val_loader, device)
        scheduler.step(v_loss)

        history['t_loss'].append(t_loss)
        history['v_loss'].append(v_loss)
        history['v_acc'].append(v_acc)
        history['v_auc'].append(v_auc)

        print(f"Эпоха {epoch:2d}/{epochs} | T-Loss: {t_loss:.4f} Acc: {t_acc:.3f} | V-Loss: {v_loss:.4f} Acc: {v_acc:.3f} AUC: {v_auc:.3f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_gw_model.pth')
            print(f"    Сохранена лучшая модель (Acc: {v_acc:.3f})")
    
    print("\n Обучение завершено.")
    
    # Сохраняем историю обучения для графиков
    np.save('training_history.npy', history)
    print(" История сохранена в training_history.npy")

if __name__ == '__main__':
    main()