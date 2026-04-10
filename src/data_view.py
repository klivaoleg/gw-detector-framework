#!/usr/bin/env python3
"""
Визуализатор датасета гравитационных волн
Поддерживает два режима:
  1. По ID из датасета: python data_view.py --id 42 --folder train
  2. По прямому пути:   python data_view.py --file sxs_datasets/bbh_test1.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Настройка шрифтов для корректного отображения кириллицы
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')

def parse_args():
    parser = argparse.ArgumentParser(description='Визуализатор датасета ГВ')
    
    # Режим 1: По ID из датасета
    parser.add_argument('--id', type=int, default=None, help='ID образца (например, 42)')
    parser.add_argument('--folder', type=str, default='train', choices=['train', 'test'], 
                       help='Папка: train или test (по умолчанию: train)')
    
    # Режим 2: Прямой путь к файлу
    parser.add_argument('--file', type=str, default=None, help='Прямой путь к CSV-файлу')
    
    parser.add_argument('--output', type=str, default=None, help='Сохранить график в файл')
    return parser.parse_args()

def load_and_visualize(filepath, source_info, output_path=None):
    """Загрузка CSV и построение графиков"""
    if not os.path.exists(filepath):
        print(f" Файл не найден: {filepath}")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    print(f"✅ Загружено {len(df)} отсчётов из: {source_info}")
    
    time = df['Time_s'].values
    det1 = df['Det1_Strain'].values
    det2 = df['Det2_Strain'].values
    
    # Безопасное извлечение метаданных (если есть)
    label = int(df['Label'].iloc[0]) if 'Label' in df.columns else -1
    snr = float(df['SNR'].iloc[0]) if 'SNR' in df.columns else 0.0
    delay = float(df['Delay_s'].iloc[0]) if 'Delay_s' in df.columns else 0.0
    time_shift = float(df['TimeShift_s'].iloc[0]) if 'TimeShift_s' in df.columns else 0.0
    
    time_ns = time * 1e9
    dt = np.mean(np.diff(time))
    freq = np.fft.fftfreq(len(time), dt)
    pos_mask = freq > 0
    f_pos = freq[pos_mask]
    
    spec1 = np.abs(np.fft.fft(det1))[pos_mask] / len(det1)
    spec2 = np.abs(np.fft.fft(det2))[pos_mask] / len(det2)
    
    correlation = np.correlate(det1 - np.mean(det1), det2 - np.mean(det2), mode='full')
    lag = np.arange(-len(det1) + 1, len(det1))
    lag_time = lag * dt * 1e9
    max_corr_idx = np.argmax(np.abs(correlation))
    measured_delay = lag_time[max_corr_idx]
    
    # Настройка фигуры
    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    ax_time1 = plt.subplot(3, 2, 1)
    ax_time2 = plt.subplot(3, 2, 3, sharex=ax_time1)
    ax_spec1 = plt.subplot(3, 2, 2)
    ax_spec2 = plt.subplot(3, 2, 4, sharex=ax_spec1)
    ax_corr = plt.subplot(3, 2, (5, 6))
    axes = [ax_time1, ax_time2, ax_spec1, ax_spec2, ax_corr]
    
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white', labelsize=9)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, color='#333333', linestyle='--', alpha=0.4)
        for spine in ax.spines.values():
            spine.set_color('#555555')
    
    status = "СИГНАЛ" if label == 1 else ("ШУМ" if label == 0 else "НЕИЗВЕСТНО")
    color_signal = '#00ffaa' if label == 1 else ('#ff6666' if label == 0 else '#aaaaaa')
    
    ax_time1.plot(time_ns, det1, color=color_signal, linewidth=0.8)
    ax_time1.set_ylabel('Амплитуда (Дет. 1)', color='#00ffff', fontsize=10)
    ax_time1.set_title(f'Временной ряд — Детектор 1 [{status}]', fontsize=11, pad=8)
    ax_time1.tick_params(axis='y', labelcolor='#00ffff')
    
    ax_time2.plot(time_ns, det2, color='#ff00ff', linewidth=0.8)
    ax_time2.set_ylabel('Амплитуда (Дет. 2)', color='#ff00ff', fontsize=10)
    ax_time2.set_xlabel('Время [нс]', color='white', fontsize=10)
    ax_time2.set_title('Временной ряд — Детектор 2', fontsize=11, pad=8)
    ax_time2.tick_params(axis='y', labelcolor='#ff00ff')
    
    ax_spec1.loglog(f_pos, spec1, color=color_signal, linewidth=1)
    ax_spec1.set_ylabel('Спектральная плотность', color='#00ffff', fontsize=10)
    ax_spec1.set_title('Спектр — Детектор 1', fontsize=11, pad=8)
    ax_spec1.tick_params(axis='y', labelcolor='#00ffff')
    
    ax_spec2.loglog(f_pos, spec2, color='#ff00ff', linewidth=1)
    ax_spec2.set_ylabel('Спектральная плотность', color='#ff00ff', fontsize=10)
    ax_spec2.set_xlabel('Частота [Гц]', color='white', fontsize=10)
    ax_spec2.set_title('Спектр — Детектор 2', fontsize=11, pad=8)
    ax_spec2.tick_params(axis='y', labelcolor='#ff00ff')
    
    ax_corr.plot(lag_time, correlation, color='#ffff00', linewidth=1)
    ax_corr.axvline(measured_delay, color='white', linestyle='--', alpha=0.7, 
                   label=f'Измеренная: {measured_delay:.3f} нс')
    ax_corr.axvline(delay * 1e9, color='#00ff00', linestyle=':', alpha=0.7, 
                   label=f'Истинная: {delay*1e9:.3f} нс')
    ax_corr.set_xlabel('Задержка [нс]', color='white', fontsize=10)
    ax_corr.set_ylabel('Значение корреляции', color='white', fontsize=10)
    ax_corr.set_title('Кросс-корреляция (оценка задержки)', fontsize=11, pad=8)
    ax_corr.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='none', labelcolor='white')
    
    # Инфо-панель
    info_text = (
        f"ИСТОЧНИК: {source_info}\n"
        f"{'='*28}\n"
        f"Класс: {status}\n"
        f"Целевой SNR: {snr:.2f}\n"
        f"Задержка (ист.): {delay*1e9:.3f} нс\n"
        f"Задержка (изм.): {measured_delay:.3f} нс\n"
        f"Сдвиг окна: {time_shift*1e9:.2f} нс\n"
        f"{'='*28}\n"
        f"Статистика:\n"
        f"   СКЗ Дет. 1: {np.sqrt(np.mean(det1**2)):.2e}\n"
        f"   СКЗ Дет. 2: {np.sqrt(np.mean(det2**2)):.2e}\n"
        f"   Коэф. корр.: {np.corrcoef(det1, det2)[0,1]:.3f}"
    )
    
    ax_time1.text(0.98, 0.98, info_text, transform=ax_time1.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', alpha=0.95, 
                          edgecolor=color_signal, linewidth=1.5),
                 color='white', fontfamily='monospace', linespacing=1.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"💾 График сохранён: {output_path}")
    else:
        plt.show()
    
    print(f"\n🔍 Сводка:")
    print(f"   Класс: {status} | SNR: {snr:.2f}")
    print(f"   Задержка: истинная={delay*1e9:.3f} нс, измеренная={measured_delay:.3f} нс")
    print(f"   Корреляция: {np.corrcoef(det1, det2)[0,1]:.3f}")

def main():
    args = parse_args()
    
    print("🔭 Визуализатор датасета гравитационных волн")
    
    # Определяем источник данных
    if args.file:
        filepath = args.file
        source_info = f"Файл: {filepath}"
    elif args.id is not None:
        filepath = f"dataset/{args.folder}/sample_{args.id}.csv"
        source_info = f"Образец #{args.id} из папки '{args.folder}'"
    else:
        print(" Ошибка: укажите либо --id, либо --file")
        print("Примеры:")
        print("  python data_view.py --id 42 --folder train")
        print("  python data_view.py --file sxs_datasets/bbh_test1.csv")
        sys.exit(1)
        
    print(f"   Источник: {source_info}")
    print("-" * 50)
    
    load_and_visualize(filepath, source_info, args.output)

if __name__ == '__main__':
    main()