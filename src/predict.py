#!/usr/bin/env python3
"""
Инференс нейросети для детектирования ГВ
Поддерживает два режима:
  1. По ID из датасета: python predict.py --id 123 --folder test
  2. По прямому пути:   python predict.py --file sxs_datasets/bbh_test1.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ===== Архитектура модели =====
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

def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("💡 Сначала запусти обучение: python train_gw_net.py")
        sys.exit(1)
    
    model = GWClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def preprocess_data(filepath):
    """Загрузка и нормализация данных (единая функция)"""
    df = pd.read_csv(filepath)
    data = df[['Det1_Strain', 'Det2_Strain']].values.T.astype(np.float64)
    
    # Нормализация (точно как при обучении)
    mean = data.mean(axis=1, keepdims=True)
    std = np.maximum(data.std(axis=1, keepdims=True), 1e-30)
    data_norm = (data - mean) / std
    
    return data_norm, df

def predict(data_norm, model, device):
    """Прогон через нейросеть"""
    tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()
    return probability

def plot_result(data_norm, probability, label_pred, confidence, status, output_path=None):
    """Визуализация результата"""
    plt.figure(figsize=(10, 4), facecolor='#0d1117')
    time_ns = np.linspace(0, len(data_norm[0]) * 2e-12 * 1e9, len(data_norm[0]))
    
    plt.plot(time_ns, data_norm[0], color='#00ffff', linewidth=0.8, label='Детектор 1')
    plt.plot(time_ns, data_norm[1], color='#ff00ff', linewidth=0.8, label='Детектор 2')
    plt.axhline(0, color='white', linestyle='--', alpha=0.3)
    
    plt.title(f'Предсказание: {status} ({confidence*100:.1f}%)', color='white')
    plt.xlabel('Время [отсчёты]', color='white')
    plt.ylabel('Норм. амплитуда', color='white')
    plt.legend(facecolor='#1a1a2e', edgecolor='none', labelcolor='white')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='#0d1117')
        print(f"💾 График сохранён: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Предсказание класса ГВ-сигнала')
    
    # Режим 1: По ID из датасета
    parser.add_argument('--id', type=int, default=None, help='ID образца в датасете')
    parser.add_argument('--folder', type=str, default='test', choices=['train', 'test'], help='Папка выборки')
    
    # 🔧 Режим 2: Прямой путь к файлу
    parser.add_argument('--file', type=str, default=None, help='Прямой путь к CSV-файлу (например, sxs_datasets/bbh.csv)')
    
    # Общие параметры
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог уверенности (0.0-1.0)')
    parser.add_argument('--output', type=str, default=None, help='Сохранить график в файл')
    parser.add_argument('--quiet', action='store_true', help='Тихий режим (без графиков)')
    
    args = parser.parse_args()

    # 🔍 Определяем путь к файлу
    if args.file:
        # Режим прямого пути
        filepath = args.file
        source_info = f"Файл: {filepath}"
    elif args.id is not None:
        # Режим по ID
        filepath = f"dataset/{args.folder}/sample_{args.id}.csv"
        source_info = f"Образец #{args.id} из папки '{args.folder}'"
    else:
        print("❌ Ошибка: укажите либо --id, либо --file")
        print("Примеры:")
        print("  python predict.py --id 123 --folder test")
        print("  python predict.py --file sxs_datasets/bbh_test1.csv")
        sys.exit(1)

    # Проверка существования файла
    if not os.path.exists(filepath):
        print(f"❌ Файл не найден: {filepath}")
        sys.exit(1)

    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('models/best_gw_model.pth', device)
    
    print(f"🔍 Анализ: {source_info}")
    
    # Предобработка и предсказание
    data_norm, df = preprocess_data(filepath)
    probability = predict(data_norm, model, device)
    
    # Интерпретация результата
    label_pred = 1 if probability >= args.threshold else 0
    status = "СИГНАЛ" if label_pred == 1 else "ШУМ"
    confidence = probability if label_pred == 1 else (1.0 - probability)
    
    # Вывод результата
    print("-" * 40)
    print(f"Вердикт сети: {status}")
    print(f"Уверенность: {confidence*100:.2f}%")
    print(f"Вероятность сигнала: {probability:.4f}")
    print("-" * 40)
    
    # Визуализация (если не тихий режим)
    if not args.quiet:
        plot_result(data_norm, probability, label_pred, confidence, status, args.output)

if __name__ == '__main__':
    main()