#!/usr/bin/env python3
"""
Конвертация данных SXS в формат для predict.py
Сохраняет результат в отдельную папку (по умолчанию sxs_datasets/)
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys

# Импортируем функцию загрузки SXS из твоего фреймворка
try:
    from sources.load_sxs import load_sxs_signal
except ImportError:
    print(" Не удалось импортировать load_sxs_signal")
    print("Проверьте корректность расположения файла")
    sys.exit(1)

def create_two_detector_data(sxs_strain, sxs_time, delay_samples=50, noise_level=1e-22):
    
    #Создаёт данные двух детекторов из одного SXS сигнала
   
    n_samples = len(sxs_strain)
    
    # Детектор 1: сигнал + шум
    noise1 = np.random.normal(0, noise_level, n_samples)
    det1 = sxs_strain + noise1
    
    # Детектор 2: сигнал со сдвигом + независимый шум
    det2 = np.zeros_like(det1)
    if delay_samples > 0:
        det2[delay_samples:] = sxs_strain[:-delay_samples]
        det2[:delay_samples] = sxs_strain[0]  # Заполняем начало значением первого отсчета
    else:
        det2[:] = sxs_strain
    
    noise2 = np.random.normal(0, noise_level, n_samples)
    det2 = det2 + noise2
    
    return det1, det2

def main():
    parser = argparse.ArgumentParser(description='Конвертация SXS в CSV для predict.py')
    
    # Основные параметры
    parser.add_argument('--sxs-id', type=str, default='SXS:BBH:0305', help='ID симуляции SXS')
    parser.add_argument('--output', type=str, default='sxs_sample.csv', help='Имя выходного CSV файла')
    
    # 🔧 Новый параметр: Папка для сохранения
    parser.add_argument('--output-dir', type=str, default='sxs_datasets', 
                       help='Папка для сохранения результатов (по умолчанию: sxs_datasets)')
    
    # Физические параметры
    parser.add_argument('--delay', type=int, default=50, help='Задержка между детекторами (отсчёты)')
    parser.add_argument('--noise', type=float, default=1e-22, help='Уровень шума')
    parser.add_argument('--label', type=int, default=1, help='Метка (1=сигнал, 0=шум)')
    parser.add_argument('--window-size', type=int, default=4096, help='Размер окна')
    
    args = parser.parse_args()

    print(f" Загрузка SXS симуляции: {args.sxs_id}")
    
    try:
        data = load_sxs_signal(args.sxs_id)
    except Exception as e:
        print(f" Ошибка загрузки SXS: {e}")
        sys.exit(1)
    
    sxs_time = data['time']
    sxs_strain = data['strain']
    
    print(f" Исходные данные: {len(sxs_strain)} отсчётов")
    
    # Находим пик сигнала
    peak_idx = np.argmax(np.abs(sxs_strain))
    
    # Вырезаем окно вокруг пика
    half_window = args.window_size // 2
    start_idx = max(0, peak_idx - half_window)
    end_idx = min(len(sxs_strain), peak_idx + half_window)
    
    # Коррекция границ
    if start_idx == 0:
        end_idx = min(len(sxs_strain), args.window_size)
    if end_idx == len(sxs_strain):
        start_idx = max(0, len(sxs_strain) - args.window_size)
    
    window_strain = sxs_strain[start_idx:end_idx]
    window_time = sxs_time[start_idx:end_idx]
    
    # Дополнение до размера окна
    if len(window_strain) < args.window_size:
        pad_size = args.window_size - len(window_strain)
        window_strain = np.pad(window_strain, (0, pad_size), mode='constant')
        window_time = np.pad(window_time, (0, pad_size), mode='edge')
    
    # Создаём два детектора
    det1, det2 = create_two_detector_data(window_strain, window_time, 
                                           delay_samples=args.delay, 
                                           noise_level=args.noise)
    
    # Создаём DataFrame
    df = pd.DataFrame({
        'Time_s': window_time,
        'Det1_Strain': det1,
        'Det2_Strain': det2,
        'Label': args.label,
        'SNR': np.max(np.abs(window_strain)) / args.noise if args.noise > 0 else 0,
        'Delay_s': args.delay * (window_time[1] - window_time[0]) if len(window_time) > 1 else 0,
        'TimeShift_s': 0.0
    })
    
    #  Логика сохранения в отдельную папку
    # 1. Создаем папку, если её нет
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Формируем полный путь: sxs_datasets/sxs_sample.csv
    full_path = os.path.join(args.output_dir, args.output)
    
    # 3. Сохраняем
    df.to_csv(full_path, index=False, float_format='%.15e')
    
    print(f"\n Готово!")
    print(f" Файл сохранён: {full_path}")
    print(f" Приблизительный SNR: {df['SNR'].iloc[0]:.2f}")
    print(f"\n Теперь можете запустить анализ:")
    print(f"   python predict.py --id custom --folder test")
    print(f"   (или просто перемести файл в нужное место)")

if __name__ == '__main__':
    main()