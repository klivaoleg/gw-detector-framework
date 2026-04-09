#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS  // Отключаем предупреждения MSVC

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <direct.h>   // Для _mkdir
#include <io.h>       // Для _access
#include <clocale>
// ================= ФИЗИЧЕСКИЕ КОНСТАНТЫ =================
const double C = 299792458.0;
const double PI = 3.14159265358979323846;


// ================= КОНФИГУРАЦИЯ ДЕТЕКТОРА =================
// Лабораторный прототип (изменяй под задачу)
const double ARM_LENGTH = 0.01;        // Длина плеча [м]
const double CAVITY_BOUNCES = 100.0;   // Усиление резонатором
const double L_EFF = ARM_LENGTH * CAVITY_BOUNCES;

const double BASELINE_M = 1.0;         // Расстояние между детекторами [м]

// Частота моделируемой ГВ (для огибающей)
// Примечание: это частота огибающей, а не несущей!
// Реальная несущая может быть 10^14 Гц, но мы моделируем только модуляцию
const double FREQ_GW = 1e10;     // 10 ГГц (частота огибающей)
const double SIGMA_T = 5e-11;          // Длительность импульса [с]

// Уровень шума (оптимистичный для будущего детектора)
const double NOISE_RMS = 1e-22;        // Целевой шум [безразмерный strain]

// ================= ПАРАМЕТРЫ СИМУЛЯЦИИ =================
const double DT = 2e-12;               // 2 пс
const int WINDOW_SIZE = 4096;          // ~8.2 нс окно
const double SIM_DURATION = WINDOW_SIZE * DT;
const int TOTAL_SAMPLES = 5000;
const double train_gear = 0.7; // количество train от total_sample
// Функция создания папки (работает везде)
bool create_dir(const char* path) {
    // Если уже существует — ок
    if (_access(path, 0) == 0) return true;
    // Пытаемся создать
    if (_mkdir(path) == 0) return true;
    // Ошибка
    std::cerr << "⚠️  Не удалось создать: " << path << "\n";
    return false;
}

// Генерация сигнала
double gw_signal(double t, double phase) {
    double envelope = std::exp(-(t * t) / (2.0 * SIGMA_T * SIGMA_T));
    return envelope * std::sin(2.0 * PI * FREQ_GW * t + phase);
}

int main() {
    // Локаль для корректного вывода чисел
    setlocale(LC_ALL, "C");
    setlocale(LC_CTYPE, "rus");
    std::cout << " Multi-Detector HFGW Dataset Generator\n";
    std::cout << "=========================================\n";
    std::cout << "Window: " << WINDOW_SIZE << " samples\n";
    std::cout << "Baseline: " << BASELINE_M << " m\n";
    std::cout << "Samples: " << TOTAL_SAMPLES << "\n\n";

    // Создаём папки (простой способ, без filesystem)
    std::cout << "Создание папок...\n";
    create_dir("dataset");
    create_dir("dataset/train");
    create_dir("dataset/test");

    // Проверка
    if (_access("dataset/train", 0) != 0) {
        std::cerr << " ОШИБКА: Не удалось создать dataset/train\n";
        return 1;
    }

    // ГСЧ
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_phase(0.0, 2.0 * PI);
    std::uniform_real_distribution<double> uniform_shift(-SIM_DURATION / 4.0, SIM_DURATION / 4.0);
    std::uniform_real_distribution<double> uniform_snr(3.0, 15.0);
    std::normal_distribution<double> noise_gen(0.0, 1.0);

    // Временная ось
    std::vector<double> t_base(WINDOW_SIZE);
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        t_base[i] = (i - WINDOW_SIZE / 2.0) * DT;
    }

    int train_count = 0;
    const int train_limit = static_cast<int>(TOTAL_SAMPLES * train_gear);

    std::cout << " Генерация...\n";

    for (int i = 0; i < TOTAL_SAMPLES; ++i) {
        bool is_signal = uniform_01(rng) > 0.5;
        int label = is_signal ? 1 : 0;

        // Направление источника
        double cos_theta = 1.0 - 2.0 * uniform_01(rng);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        double phi = 2.0 * PI * uniform_01(rng);
        double n_x = sin_theta * std::cos(phi);
        double delay = (BASELINE_M * n_x) / C;

        // Параметры
        double time_shift = uniform_shift(rng);
        double phase = uniform_phase(rng);
        double target_snr = is_signal ? uniform_snr(rng) : 0.0;

        std::vector<double> det1(WINDOW_SIZE, 0.0);
        std::vector<double> det2(WINDOW_SIZE, 0.0);

        if (is_signal) {
            // Расчёт RMS для нормализации
            std::vector<double> raw1(WINDOW_SIZE), raw2(WINDOW_SIZE);
            double sum_sq = 0.0;
            for (int j = 0; j < WINDOW_SIZE; ++j) {
                double t = t_base[j] - time_shift;
                raw1[j] = gw_signal(t, phase);
                raw2[j] = gw_signal(t - delay, phase);
                sum_sq += raw1[j] * raw1[j];
            }

            double signal_rms = std::sqrt(sum_sq / WINDOW_SIZE);
            double scale = (signal_rms > 1e-30) ? (target_snr * NOISE_RMS / signal_rms) : 0.0;

            // Сигнал + независимый шум
            for (int j = 0; j < WINDOW_SIZE; ++j) {
                det1[j] = raw1[j] * scale + noise_gen(rng) * NOISE_RMS;
                det2[j] = raw2[j] * scale + noise_gen(rng) * NOISE_RMS;
            }
        }
        else {
            // Только шум
            for (int j = 0; j < WINDOW_SIZE; ++j) {
                det1[j] = noise_gen(rng) * NOISE_RMS;
                det2[j] = noise_gen(rng) * NOISE_RMS;
            }
        }

        // Запись CSV
        std::string folder = (i < train_limit) ? "dataset/train/" : "dataset/test/";
        std::string filename = folder + "sample_" + std::to_string(i) + ".csv";

        std::ofstream csv(filename);
        if (!csv.is_open()) {
            std::cerr << " Ошибка записи: " << filename << "\n";
            continue;
        }

        csv << std::scientific << std::setprecision(12);
        csv << "Time_s,Det1_Strain,Det2_Strain,Label,SNR,Delay_s,TimeShift_s\n";
        for (int j = 0; j < WINDOW_SIZE; ++j) {
            csv << t_base[j] << ","
                << det1[j] << ","
                << det2[j] << ","
                << label << ","
                << target_snr << ","
                << delay << ","
                << time_shift << "\n";
        }
        csv.close();

        if (i < train_limit) train_count++;

        if (i % 10 == 0) {
            std::cout << " " << i << "/" << TOTAL_SAMPLES << "\n";
        }
    }

    std::cout << "\n ГОТОВО!\n";
    std::cout << "Train: " << train_count << " файлов\n";
    std::cout << " Test:  " << (TOTAL_SAMPLES - train_count) << " файлов\n";
    std::cout << " Данные в папке: dataset/\n";

    // Чтобы окно не закрылось
    std::cout << "\n Нажмите Enter для выхода...";
    std::cin.get();

    return 0;
}