import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks, welch, butter, filtfilt
import sys
from datetime import datetime

def advanced_motor_diagnostic(file_path, known_rpm=None, min_freq=20, max_freq=2000):
    """
    Продвинутая диагностика мотора с учетом реальных оборотов.
    Результаты сохраняются в PNG и TXT файлы.
    """
    
    # Создаем список для сбора текстового отчета
    report_lines = []
    
    def log(message):
        """Функция для одновременного вывода в консоль и сохранения в отчет"""
        print(message)
        report_lines.append(message)
    
    # Заголовок отчета
    log("="*70)
    log("ДИАГНОСТИКА ЭЛЕКТРОДВИГАТЕЛЯ МЕТОДОМ ФУРЬЕ-АНАЛИЗА")
    log("="*70)
    log(f"Дата и время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Анализируемый файл: {file_path}")
    
    # 1. Загрузка и предобработка
    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        log(f"Ошибка загрузки файла: {e}")
        return
    
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Нормализация
    data = data / np.max(np.abs(data))
    duration = len(data) / sample_rate
    
    log(f"Длительность записи: {duration:.2f} сек")
    log(f"Частота дискретизации: {sample_rate} Гц")
    log(f"Диапазон анализа: {min_freq} - {max_freq} Гц")
    
    # 2. Полосовой фильтр для удаления шумов
    nyquist = sample_rate * 0.5
    low = min_freq / nyquist
    high = min(max_freq / nyquist, 0.99)
    b, a = butter(4, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    # 3. Усредненный спектр с высоким разрешением
    nperseg = 8192
    f, Pxx = welch(filtered_data, fs=sample_rate, nperseg=nperseg, 
                   noverlap=nperseg//2, scaling='density')
    
    mask = (f >= min_freq) & (f <= max_freq)
    f = f[mask]
    Pxx = Pxx[mask]
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    
    # 4. Поиск пиков с адаптивным порогом
    window = len(f) // 20
    rolling_mean = np.convolve(Pxx_db, np.ones(window)/window, mode='same')
    threshold = rolling_mean + 6
    
    peaks, _ = find_peaks(Pxx_db, height=threshold, distance=10)
    
    # 5. Определение основной частоты (RPM)
    fundamental_candidates = []
    for p in peaks:
        if f[p] < 300:
            has_harmonics = False
            for mult in [2, 3, 4]:
                harmonic_freq = f[p] * mult
                if harmonic_freq > max_freq:
                    break
                closest_peak_idx = np.argmin(np.abs(f - harmonic_freq))
                if abs(f[closest_peak_idx] - harmonic_freq) < f[p] * 0.05:
                    if Pxx_db[closest_peak_idx] > Pxx_db[p] - 20:
                        has_harmonics = True
                        break
            if has_harmonics or known_rpm:
                fundamental_candidates.append((f[p], Pxx_db[p], has_harmonics))
    
    # Выбор основной частоты
    if known_rpm:
        fundamental_freq = known_rpm / 60.0
        log(f"\nЗаданы известные обороты: {known_rpm} об/мин ({fundamental_freq:.2f} Гц)")
    elif fundamental_candidates:
        fundamental_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # ========== ВОТ ЭТИ 5 СТРОК ==========
        # Проверка на паттерн "критический дисбаланс" (1X и 2X почти равны)
        if len(fundamental_candidates) >= 2:
            f1, f2 = fundamental_candidates[0][0], fundamental_candidates[1][0]
            if abs(max(f1, f2) / min(f1, f2) - 2.0) < 0.1:
                if abs(fundamental_candidates[0][1] - fundamental_candidates[1][1]) < 3:
                    fundamental_freq = min(f1, f2)
                    log(f"\n⚠️ ОБНАРУЖЕН ПАТТЕРН КРИТИЧЕСКОГО ДИСБАЛАНСА!")
                    log(f"   Выбрана меньшая частота как 1X: {fundamental_freq:.2f} Гц")
                else:
                    fundamental_freq = fundamental_candidates[0][0]
            else:
                fundamental_freq = fundamental_candidates[0][0]
        else:
            fundamental_freq = fundamental_candidates[0][0]
        # =====================================
        
        rpm_est = fundamental_freq * 60
        log(f"Частота вращения принята: {fundamental_freq:.2f} Гц ({rpm_est:.0f} об/мин)")
    else:
        low_peaks = peaks[f[peaks] < 200]
        if len(low_peaks) > 0:
            strongest = low_peaks[np.argmax(Pxx_db[low_peaks])]
            fundamental_freq = f[strongest]
            rpm_est = fundamental_freq * 60
            log(f"\nПредположительная частота вращения: {fundamental_freq:.2f} Гц ({rpm_est:.0f} об/мин)")
            log("ВНИМАНИЕ: Гармоники не обнаружены, возможна электрическая неисправность!")
        else:
            fundamental_freq = None
            log("\nНЕ УДАЛОСЬ ОПРЕДЕЛИТЬ ЧАСТОТУ ВРАЩЕНИЯ!")
            log("Вероятные причины:")
            log("  - Короткое замыкание или искрение (хаотичный шум)")
            log("  - Двигатель не вращается")
            log("  - Слишком тихая запись или сильные помехи")
    
    # 6. Детальный анализ пиков
    log("\n" + "="*70)
    log("СПЕКТРАЛЬНЫЙ АНАЛИЗ")
    log("="*70)
    
    log(f"\nОбнаружено пиков: {len(peaks)}")
    log("\n10 самых мощных частотных составляющих:")
    sorted_peaks = sorted(zip(f[peaks], Pxx_db[peaks]), key=lambda x: x[1], reverse=True)
    for i, (freq, level) in enumerate(sorted_peaks[:10], 1):
        log(f"  {i:2}. {freq:8.2f} Гц  |  Уровень: {level:6.2f} dB")
    
    if fundamental_freq:
        rpm = fundamental_freq * 60
        
        log("\n" + "="*70)
        log("АНАЛИЗ ГАРМОНИК ОСНОВНОЙ ЧАСТОТЫ")
        log("="*70)
        log(f"\nБазовая частота (1X): {fundamental_freq:.2f} Гц ({rpm:.0f} об/мин)\n")
        
        harmonics_analysis = []
        log("Множитель | Ожидаемая  | Фактическая | Уровень  | Отклонение | Значимость")
        log("-"*70)
        
        for mult in [0.5, 1, 2, 3, 4, 5, 6, 7, 8]:
            target = fundamental_freq * mult
            if target > max_freq:
                break
            
            distances = np.abs(f[peaks] - target)
            closest_idx = peaks[np.argmin(distances)]
            closest_freq = f[closest_idx]
            deviation = abs(closest_freq - target) / target * 100
            level = Pxx_db[closest_idx]
            
            if deviation < 5 and level > -80:
                severity = "ВЫСОКАЯ" if level > -60 else "СРЕДНЯЯ" if level > -70 else "НИЗКАЯ"
                harmonics_analysis.append((mult, target, closest_freq, level, deviation, severity))
                log(f"   {mult:4.1f}X   | {target:8.2f}   | {closest_freq:8.2f}    | {level:6.2f} |    {deviation:5.2f}%   | {severity}")
            elif deviation < 10:
                log(f"   {mult:4.1f}X   | {target:8.2f}   | {closest_freq:8.2f}    | {level:6.2f} |    {deviation:5.2f}%   | СЛАБЫЙ")
            else:
                log(f"   {mult:4.1f}X   | {target:8.2f}   |     --     |   --   |     --    | ОТСУТСТВУЕТ")
        
                # Диагностические заключения
        log("\n" + "="*70)
        log("ДИАГНОСТИЧЕСКОЕ ЗАКЛЮЧЕНИЕ")
        log("="*70)
        
        issues_found = []
        max_severity_score = 0  # 0 - нет проблем, 1 - низкая, 2 - средняя, 3 - критическая
        
        # Проверяем дробные гармоники (КРИТИЧЕСКИЙ ДЕФЕКТ)
        fractional_issues = [h for h in harmonics_analysis if h[0] in [0.5, 1.5, 2.5, 3.5]]
        if fractional_issues:
            issues_found.append("ДРОБНЫЕ ГАРМОНИКИ (КРИТИЧЕСКИЙ)")
            max_severity_score = max(max_severity_score, 3)
            log("\n🔴 КРИТИЧЕСКИЙ ДЕФЕКТ: ОБНАРУЖЕНЫ ДРОБНЫЕ ГАРМОНИКИ")
            log("   Это указывает на серьезные механические повреждения:")
            log("   • Трещина вала ротора")
            log("   • Износ шпонки или шлицевого соединения")
            log("   • Люфт в подшипнике скольжения")
            log("   • Ослабление посадки подшипника на валу")
            log("   ⚠ ОСТОРОЖНО: ВОЗМОЖЕН КРИТИЧЕСКИЙ ДИСБАЛАНС")
            log("   ❌ РЕКОМЕНДАЦИЯ: Немедленная остановка и ревизия мотора! ")
            log("   ❌ РЕКОМЕНДАЦИЯ: Cначала устранить дисбаланс и повторить анализ. ")
            log("   Разница амплитуд (1X-2X) > 15 dB-Норма ; 8 - 15 dB - Умеренный; < 8 dB - Сильный ")
            
        
        # Проверяем высокие гармоники (СРЕДНИЙ ДЕФЕКТ)
        high_harmonics = [h for h in harmonics_analysis if h[0] in [2, 3, 4] and h[5] in ["ВЫСОКАЯ", "СРЕДНЯЯ"]]
        if high_harmonics:
                    
            # ========== ВОТ ЭТИ 3 СТРОКИ ==========
            # Если 2X почти равна 1X или громче — это КРИТИЧЕСКИЙ дисбаланс
            harmonic_2x = next((h for h in high_harmonics if h[0] == 2), None)
            if harmonic_2x:
                level_1x = next(h[3] for h in harmonics_analysis if h[0] == 1)
                level_2x = harmonic_2x[3]
                if level_1x - level_2x < 3:  # Разница < 3 dB
                    max_severity_score = 3
                    log("\n🔴 КРИТИЧЕСКИЙ ДИСБАЛАНС: 2X гармоника почти равна 1X!")
                    log(f"   Разница 1X-2X = {level_1x - level_2x:.1f} dB")
            # =====================================
            issues_found.append("УСИЛЕННЫЕ ГАРМОНИКИ (СРЕДНИЙ)")
            max_severity_score = max(max_severity_score, 2)
            log("\n🟡 СРЕДНИЙ ДЕФЕКТ: УСИЛЕННЫЕ ВЫСОКИЕ ГАРМОНИКИ")
            if any(h[0] == 2 for h in high_harmonics):
                log("   • 2X гармоника усилена:")
                log("     - Дисбаланс ротора")
                log("     - Несоосность валов")
                log("     - Ослабление крепления мотора к раме")
                log("     ⚠️ РЕКОМЕНДАЦИЯ: Проверить балансировку и соосность при плановом ТО")
            if any(h[0] == 3 for h in high_harmonics):
                log("   • 3X гармоника усилена:")
                log("     - Износ муфты или неправильная центровка")
                log("     - Проблемы с лопастями вентилятора")
                log("     ⚠️ РЕКОМЕНДАЦИЯ: Проверить соединительную муфту")
            if any(h[0] == 4 for h in high_harmonics):
                log("   • 4X гармоника усилена:")
                log("     - Дефект подшипника качения (частота тел качения)")
                log("     - Проблемы с ременной передачей")
                log("     ⚠️ РЕКОМЕНДАЦИЯ: Заменить подшипники при ближайшем обслуживании")
        
        # Анализ подшипниковых частот (НИЗКИЙ ДЕФЕКТ)
        bearing_peaks = peaks[(f[peaks] > 500) & (f[peaks] < 2000)]
        if len(bearing_peaks) > 5:
            non_harmonic = []
            for bp in bearing_peaks[:10]:
                ratio = f[bp] / fundamental_freq
                if abs(ratio - round(ratio)) > 0.1:
                    non_harmonic.append(f[bp])
            
            if non_harmonic:
                issues_found.append("ПОДШИПНИКОВЫЕ ЧАСТОТЫ (НИЗКИЙ)")
                max_severity_score = max(max_severity_score, 1)
                log("\n🟢 НЕЗНАЧИТЕЛЬНЫЙ ДЕФЕКТ: ПОДШИПНИКОВЫЕ ШУМЫ")
                log("   Обнаружены высокочастотные негармонические пики:")
                log("   • Возможен начальный износ подшипников качения")
                log("   • Небольшой недостаток смазки")
                log(f"   Подозрительные частоты: {', '.join([f'{nf:.0f}' for nf in non_harmonic[:5]])} Гц")
                log("   ℹ️ РЕКОМЕНДАЦИЯ: Контроль состояния при следующем ТО")
        
        if not issues_found:
            max_severity_score = 0
            log("\n✅ ЗНАЧИМЫХ ОТКЛОНЕНИЙ НЕ ОБНАРУЖЕНО")
            log("   Спектр вибрации в пределах нормы.")
            log("   Двигатель, вероятно, исправен.")
        
        # ИТОГОВАЯ ОЦЕНКА СОСТОЯНИЯ (по максимальному весу)
        log("\n" + "="*70)
        log("ИТОГОВАЯ ОЦЕНКА СОСТОЯНИЯ")
        log("="*70)
        
        if max_severity_score == 3:
            log("\n🔴 КРИТИЧЕСКОЕ СОСТОЯНИЕ")
            log("   Обнаружены дефекты, угрожающие целостности оборудования.")
            log("   ❌ ЭКСПЛУАТАЦИЯ ЗАПРЕЩЕНА!")
            log("   Требуется немедленная остановка и ремонт.")
        elif max_severity_score == 2:
            log("\n🟡 ТРЕБУЕТ ВНИМАНИЯ")
            log("   Обнаружены отклонения от нормы средней тяжести.")
            log("   ⚠️ Рекомендуется провести обслуживание в течение недели.")
        elif max_severity_score == 1:
            log("\n🟢 УДОВЛЕТВОРИТЕЛЬНОЕ")
            log("   Обнаружены незначительные отклонения.")
            log("   ✅ Можно продолжать эксплуатацию с периодическим контролем.")
        else:
            log("\n🟢 ХОРОШЕЕ")
            log("   Значимых дефектов не обнаружено.")
            log("   ✅ Двигатель исправен.")
    
    # 7. Визуализация с аннотациями
    plt.figure(figsize=(16, 10))
    
    # Спектрограмма
    plt.subplot(2, 1, 1)
    f_spec, t_spec, Sxx = spectrogram(filtered_data, fs=sample_rate, nperseg=2048, noverlap=1536)
    mask_spec = (f_spec >= min_freq) & (f_spec <= max_freq)
    plt.pcolormesh(t_spec, f_spec[mask_spec], 10*np.log10(Sxx[mask_spec,:] + 1e-12), 
                   shading='gouraud', cmap='viridis')
    plt.colorbar(label='dB')
    plt.ylabel('Частота (Гц)')
    plt.xlabel('Время (с)')
    plt.title(f'Спектрограмма: {file_path}')
    if fundamental_freq:
        for mult in [1, 2, 3, 4]:
            plt.axhline(y=fundamental_freq*mult, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Детальный спектр
    plt.subplot(2, 1, 2)
    plt.semilogy(f, 10**(Pxx_db/10), 'b-', alpha=0.7, linewidth=1, label='Спектр')
    plt.scatter(f[peaks], 10**(Pxx_db[peaks]/10), c='red', s=50, alpha=0.8, zorder=5, label='Пики')
    
    if fundamental_freq:
        for mult in np.arange(0.5, 9, 0.5):
            plt.axvline(x=fundamental_freq*mult, color='green', linestyle=':', alpha=0.4)
        plt.axvline(x=fundamental_freq, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'1X = {fundamental_freq:.1f} Гц')
    
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Мощность (Вт)')
    plt.title('Спектр мощности с обнаруженными пиками')
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim(min_freq, max_freq)
    plt.legend()
    
    plt.tight_layout()
    
    # Сохранение графиков и отчета
    base_name = file_path.replace('.wav', '').replace('.WAV', '')
    png_file = f"{base_name}_diagnostic.png"
    txt_file = f"{base_name}_report.txt"
    
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сохранение текстового отчета
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    log(f"\n{'='*70}")
    log(f"РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
    log(f"  График: {png_file}")
    log(f"  Отчет:  {txt_file}")
    log("="*70)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        known_rpm = int(sys.argv[2]) if len(sys.argv) > 2 else None
        advanced_motor_diagnostic(file_path, known_rpm)
    else:
        print("Использование: python motor_diag.py файл.wav [обороты]")
        print("Пример: python motor_diag.py motor.wav 3000")
        print("\nРезультаты будут сохранены в:")
        print("  - файл_wav_diagnostic.png (график)")
        print("  - файл_wav_report.txt (текстовый отчет)")
