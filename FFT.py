import sys
import os
import time
import numpy as np
from scipy import fft
from scipy import signal
from scipy.io import wavfile
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QFileDialog, QComboBox, 
                             QLineEdit, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

def generate_analytical_data(filename, sample_rate, duration):
    """
    构造测试用高精度信号文件，生成双通道基准验证数据以供多维架构校验。
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float64)
    # 通道 1: 50Hz + 120Hz
    ch1 = np.sin(2 * np.pi * 50 * t, dtype=np.float64) + \
          0.5 * np.sin(2 * np.pi * 120 * t, dtype=np.float64) + \
          0.1 * np.random.normal(size=t.shape).astype(np.float64)
    # 通道 2: 80Hz + 200Hz
    ch2 = np.sin(2 * np.pi * 80 * t, dtype=np.float64) + \
          0.3 * np.sin(2 * np.pi * 200 * t, dtype=np.float64) + \
          0.1 * np.random.normal(size=t.shape).astype(np.float64)
          
    composite_signal = np.column_stack((ch1, ch2))
    with open(filename, 'wb') as file_handler:
        file_handler.write(composite_signal.tobytes())

class DataFormatConverter:
    """
    数据格式自适应解析引擎。
    执行多通道提取、采样率自整定，并将各类外部文件标准化为双精度浮点的交织二进制缓存流。
    """
    @staticmethod
    def convert_to_bin_cache(input_file, cache_file, default_sr):
        ext = os.path.splitext(input_file)[1].lower()
        sample_rate = default_sr
        num_channels = 1
        channel_names = ["Channel 1"]
        
        try:
            if ext == '.wav':
                sr, data = wavfile.read(input_file)
                sample_rate = float(sr)
                if data.dtype == np.int16:
                    data = data.astype(np.float64) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float64) / 2147483648.0
                else:
                    data = data.astype(np.float64)
                    
                if len(data.shape) > 1:
                    num_channels = data.shape[1]
                    channel_names = [f"Audio_Ch_{i+1}" for i in range(num_channels)]
                else:
                    data = data.reshape(-1, 1)
                    
                with open(cache_file, 'wb') as f:
                    f.write(data.tobytes())
            
            elif ext == '.csv':
                with open(input_file, 'r', encoding='utf-8') as f:
                    header_line = f.readline().strip()
                headers = [h.strip('"').strip() for h in header_line.split(',')]
                
                data = np.loadtxt(input_file, delimiter=',', skiprows=1, dtype=np.float64)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                if 'time' in headers:
                    time_idx = headers.index('time')
                    time_data = data[:, time_idx]
                    dt = np.mean(np.diff(time_data))
                    if dt > 0:
                        sample_rate = 1.0 / dt
                    
                    channels_data = np.delete(data, time_idx, axis=1)
                    channel_names = [h for i, h in enumerate(headers) if i != time_idx]
                else:
                    channels_data = data
                    channel_names = [f"CSV_Col_{i+1}" for i in range(channels_data.shape[1])]
                    
                num_channels = channels_data.shape[1]
                with open(cache_file, 'wb') as f:
                    f.write(channels_data.tobytes())
                    
            elif ext in ['.bin', '.dat']:
                data = np.fromfile(input_file, dtype=np.float64).reshape(-1, 1)
                with open(cache_file, 'wb') as f:
                    f.write(data.tobytes())
            else:
                raise ValueError(f"不支持的文件扩展名: {ext}")
                
            return True, sample_rate, num_channels, channel_names, ""
        except Exception as e:
            return False, default_sr, 1, [""], str(e)

class FFTComputeThread(QThread):
    """
    支持多通道并发矩阵运算与双模态参数配置的独立计算线程。
    """
    update_signal = pyqtSignal(float, list, list, object, object)
    finished_signal = pyqtSignal()

    def __init__(self, filename, sample_rate, num_channels, mode):
        super().__init__()
        self.filename = filename
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.mode = mode
        self._is_running = True
        
        if self.mode == "FAST":
            self.chunk_size = 1024
            self.overlap_ratio = 0.25
            self.window = signal.windows.hamming(self.chunk_size, sym=False).astype(np.float64)
            self.n_fft = self.chunk_size
        else: # HIGH_PRECISION
            self.chunk_size = 4096
            self.overlap_ratio = 0.85
            self.window = signal.windows.blackmanharris(self.chunk_size, sym=False).astype(np.float64)
            self.n_fft = self.chunk_size * 4 

    def run(self):
        step_size = int(self.chunk_size * (1 - self.overlap_ratio))
        file_size = os.path.getsize(self.filename)
        total_frames = (file_size // 8) // self.num_channels 
        
        with open(self.filename, 'rb') as file_handler:
            current_pos = 0
            
            while current_pos + self.chunk_size <= total_frames and self._is_running:
                file_handler.seek(current_pos * self.num_channels * 8)
                raw_bytes = file_handler.read(self.chunk_size * self.num_channels * 8)
                
                if not raw_bytes or len(raw_bytes) < self.chunk_size * self.num_channels * 8:
                    break
                
                data_chunk = np.frombuffer(raw_bytes, dtype=np.float64).reshape(self.chunk_size, self.num_channels)
                
                window_2d = self.window[:, np.newaxis]
                windowed_data = data_chunk * window_2d
                
                fft_complex_result = fft.fft(windowed_data, n=self.n_fft, axis=0)
                fft_frequencies = fft.fftfreq(self.n_fft, 1.0 / self.sample_rate)
                
                window_coherent_gain = np.sum(self.window)
                magnitude_spectrum = np.abs(fft_complex_result) / window_coherent_gain
                magnitude_spectrum[1:] = 2 * magnitude_spectrum[1:]
                
                positive_freq_mask = fft_frequencies >= 0
                valid_freqs = fft_frequencies[positive_freq_mask]
                valid_mags = magnitude_spectrum[positive_freq_mask, :] 
                
                dominant_indices = np.argmax(valid_mags, axis=0)
                dominant_frequencies = valid_freqs[dominant_indices]
                dominant_magnitudes = [valid_mags[idx, i] for i, idx in enumerate(dominant_indices)]
                
                current_time = current_pos / self.sample_rate
                
                self.update_signal.emit(
                    current_time, 
                    list(dominant_frequencies), 
                    list(dominant_magnitudes), 
                    np.copy(valid_freqs), 
                    np.copy(valid_mags)
                )
                
                current_pos += step_size
                
                if self.mode == "FAST":
                    time.sleep(step_size / self.sample_rate)
                else:
                    time.sleep(0.01)
                
        self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.wait()

class MultiModeSpectrumAnalyzerGUI(QMainWindow):
    """
    多通道图形主控界面。
    """
    def __init__(self):
        super().__init__()
        self.cache_file = "system_data_cache.bin"
        self.current_sample_rate = 2000.0
        self.current_num_channels = 1
        self.current_channel_names = ["Default"]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("多维度多模态高精度频谱分析系统 (SciPy Backend)")
        self.resize(1100, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("导入外部数据文件")
        self.btn_load.clicked.connect(self.load_external_file)
        control_layout.addWidget(self.btn_load)
        
        self.btn_generate = QPushButton("生成系统校准数据")
        self.btn_generate.clicked.connect(self.generate_baseline_data)
        control_layout.addWidget(self.btn_generate)
        
        control_layout.addWidget(QLabel("默认采样率(Hz):"))
        self.sr_input = QLineEdit("2000")
        self.sr_input.setMaximumWidth(80)
        control_layout.addWidget(self.sr_input)
        
        control_layout.addWidget(QLabel("分析模态:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["动态快响应模式", "静态高精度模式"])
        control_layout.addWidget(self.mode_combo)
        
        self.btn_start = QPushButton("启动分析序列")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setEnabled(False)
        control_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("中止")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)
        
        main_layout.addLayout(control_layout)
        
        self.status_label = QLabel("系统就绪。请导入包含 time 及独立通道属性的 csv 文件。")
        self.status_label.setFont(self._get_monospace_font())
        main_layout.addWidget(self.status_label)
        
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("多通道瞬态频域分布阵列", size="14pt")
        self.plot_widget.setLabel('left', "绝对幅值")
        self.plot_widget.setLabel('bottom', "频率", units="Hz")
        self.plot_widget.showGrid(x=True, y=True)
        main_layout.addWidget(self.plot_widget)
        
        self.thread = None

    def _get_monospace_font(self):
        font = self.status_label.font()
        font.setPointSize(11)
        font.setFamily("Consolas")
        return font

    def load_external_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入数据文件", "", "Data Files (*.wav *.csv *.bin *.dat);;All Files (*)"
        )
        if file_path:
            try:
                default_sr = float(self.sr_input.text())
            except ValueError:
                default_sr = 2000.0
                
            self.status_label.setText("正在执行格式转换与阵列映射...")
            QApplication.processEvents()
            
            success, sr, num_ch, ch_names, err = DataFormatConverter.convert_to_bin_cache(file_path, self.cache_file, default_sr)
            if success:
                self.current_sample_rate = sr
                self.current_num_channels = num_ch
                self.current_channel_names = ch_names
                self.sr_input.setText(f"{sr:.2f}")
                self.status_label.setText(f"数据装载完毕: {os.path.basename(file_path)} | 解析采样率: {sr:.2f} Hz | 有效解析通道数: {num_ch}")
                self.btn_start.setEnabled(True)
            else:
                QMessageBox.critical(self, "解析失败", f"文件解析异常:\n{err}")
                self.status_label.setText("系统就绪。")

    def generate_baseline_data(self):
        try:
            self.current_sample_rate = float(self.sr_input.text())
        except ValueError:
            self.current_sample_rate = 2000.0
        
        self.status_label.setText("正在合成双精度基准校验数据(双通道)...")
        QApplication.processEvents()
        generate_analytical_data(self.cache_file, self.current_sample_rate, duration=10)
        self.current_num_channels = 2
        self.current_channel_names = ["Ch1_50Hz_120Hz", "Ch2_80Hz_200Hz"]
        self.status_label.setText(f"校验数据生成完毕 | 采样率: {self.current_sample_rate} Hz | 通道数: 2")
        self.btn_start.setEnabled(True)

    def start_analysis(self):
        self.btn_start.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.plot_widget.clear()
        if hasattr(self, 'legend') and self.legend is not None:
            self.legend.scene().removeItem(self.legend)
        self.legend = self.plot_widget.addLegend()
        
        self.spectrum_curves = []
        colors = [(0, 114, 189), (217, 83, 25), (237, 177, 32), (126, 47, 142), (119, 172, 48)]
        for i, name in enumerate(self.current_channel_names):
            color = colors[i % len(colors)]
            curve = self.plot_widget.plot(pen=pg.mkPen(color=color, width=2), name=name)
            self.spectrum_curves.append(curve)
        
        mode_str = "FAST" if self.mode_combo.currentIndex() == 0 else "HIGH_PRECISION"
        
        self.thread = FFTComputeThread(
            filename=self.cache_file,
            sample_rate=self.current_sample_rate,
            num_channels=self.current_num_channels,
            mode=mode_str
        )
        self.thread.update_signal.connect(self.update_gui)
        self.thread.finished_signal.connect(self.on_analysis_finished)
        self.thread.start()

    def stop_analysis(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        self.on_analysis_finished()

    def update_gui(self, current_time, dom_freqs, dom_mags, freqs, mags):
        freq_str = ", ".join([f"{f:0.1f}" for f in dom_freqs])
        mag_str = ", ".join([f"{m:0.3f}" for m in dom_mags])
        status_text = (f"时间基准: {current_time:06.3f}s | "
                       f"阵列主频(Hz): [{freq_str}] | "
                       f"绝对幅值: [{mag_str}]")
        self.status_label.setText(status_text)
        
        for i, curve in enumerate(self.spectrum_curves):
            if i < mags.shape[1]:
                curve.setData(freqs, mags[:, i])

    def on_analysis_finished(self):
        self.status_label.setText(self.status_label.text() + " [计算序列挂起]")
        self.btn_start.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except OSError:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MultiModeSpectrumAnalyzerGUI()
    main_window.show()
    sys.exit(app.exec())