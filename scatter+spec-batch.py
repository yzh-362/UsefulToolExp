import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import simpson
from scipy import sparse
from scipy.sparse.linalg import spsolve
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QInputDialog, QSpinBox, QDoubleSpinBox, 
                             QGroupBox, QSplitter, QDialog, QTableWidget, 
                             QTableWidgetItem, QComboBox, QFormLayout, 
                             QDialogButtonBox, QMessageBox, QHeaderView, 
                             QScrollArea, QTabWidget, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 核心算法与工具函数 ====================

def generate_mock_spectrum_data(filename):
    """生成带有基线和噪声的多个高斯峰模拟数据"""
    x = np.linspace(0, 1000, 2000)
    y = 50 * np.exp(-((x - 250) / 20)**2) + \
        120 * np.exp(-((x - 500) / 40)**2) + \
        80 * np.exp(-((x - 750) / 30)**2) + \
        np.random.normal(0, 2, len(x)) + \
        0.05 * x
    
    df = pd.DataFrame({'Wavenumber': x, 'Intensity': y})
    df.to_csv(filename, index=False)
    return filename

def baseline_arpls(x, y, lam=1e6, niter=15, max_points=2000):
    """基于非对称加权惩罚最小二乘法 (arPLS) 的自适应基线扣除"""
    L = len(y)
    if L > max_points:
        block_size = L // max_points
        x_down = []
        y_down = []
        for i in range(0, L, block_size):
            block_x = x[i:i+block_size]
            block_y = y[i:i+block_size]
            if len(block_y) > 0:
                min_idx = np.argmin(block_y)
                x_down.append(block_x[min_idx])
                y_down.append(block_y[min_idx])
        x_down = np.array(x_down)
        y_down = np.array(y_down)
    else:
        x_down = x
        y_down = y

    L_down = len(y_down)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L_down, L_down - 2), dtype=np.float64)
    D_matrix = lam * D.dot(D.transpose())
    w = np.ones(L_down)
    z_down = np.copy(y_down)

    for i in range(niter):
        W = sparse.spdiags(w, 0, L_down, L_down)
        Z = W + D_matrix
        try:
            z_down = spsolve(Z.tocsc(), w * y_down)
        except Exception:
            break
        
        d = y_down - z_down
        d_negative = d[d < 0]
        if len(d_negative) > 0:
            m = np.mean(d_negative)
            s = np.std(d_negative)
        else:
            m = 0.0
            s = 0.01
            
        exponent = np.clip(2.0 * (d - (2 * s - m)) / (s + 1e-6), -700, 700)
        w = 1.0 / (1.0 + np.exp(exponent))

    if L > max_points:
        z = np.interp(x, x_down, z_down)
    else:
        z = z_down

    return z

# ==================== 共享组件与基类 ====================

class DataImportDialog(QDialog):
    """统一的数据导入配置组件（含鲁棒性约束机制）"""
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.setWindowTitle(f"自定义数据导入设置 - {os.path.basename(filepath)}")
        self.resize(600, 400)
        
        self.delimiter_map = {
            "逗号 (,)": ",",
            "制表符 (Tab)": "\t",
            "空格 (Space)": "\\s+",
            "分号 (;)": ";"
        }
        
        self.final_x = None
        self.final_y = None
        
        self.init_ui()
        self.update_preview()

    def init_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.combo_sep = QComboBox()
        self.combo_sep.addItems(list(self.delimiter_map.keys()))
        self.combo_sep.currentIndexChanged.connect(self.update_preview)
        form_layout.addRow("数据分隔符:", self.combo_sep)
        
        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(0, 1000)
        self.spin_skip.setValue(0)
        self.spin_skip.valueChanged.connect(self.update_preview)
        form_layout.addRow("跳过表头行数:", self.spin_skip)
        
        self.spin_x_col = QSpinBox()
        self.spin_x_col.setRange(0, 100)
        self.spin_x_col.setValue(0)
        self.spin_x_col.valueChanged.connect(self.update_preview)
        form_layout.addRow("X轴对应列索引:", self.spin_x_col)
        
        self.spin_y_col = QSpinBox()
        self.spin_y_col.setRange(0, 100)
        self.spin_y_col.setValue(1)
        self.spin_y_col.valueChanged.connect(self.update_preview)
        form_layout.addRow("Y轴对应列索引:", self.spin_y_col)
        
        layout.addLayout(form_layout)
        
        layout.addWidget(QLabel("数据预览 (仅显示前30行有效数据):"))
        self.table_preview = QTableWidget()
        self.table_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_preview)
        
        self.btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_box.accepted.connect(self.accept_data)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

    def update_preview(self):
        sep_key = self.combo_sep.currentText()
        sep = self.delimiter_map[sep_key]
        skiprows = self.spin_skip.value()
        try:
            df_preview = pd.read_csv(self.filepath, sep=sep, skiprows=skiprows, nrows=30, engine='python', header=None)
            self.table_preview.setRowCount(df_preview.shape[0])
            self.table_preview.setColumnCount(df_preview.shape[1])
            headers = []
            for col in range(df_preview.shape[1]):
                if col == self.spin_x_col.value():
                    headers.append(f"列 {col} (X轴)")
                elif col == self.spin_y_col.value():
                    headers.append(f"列 {col} (Y轴)")
                else:
                    headers.append(f"列 {col}")
            self.table_preview.setHorizontalHeaderLabels(headers)
            for row in range(df_preview.shape[0]):
                for col in range(df_preview.shape[1]):
                    val = str(df_preview.iloc[row, col])
                    self.table_preview.setItem(row, col, QTableWidgetItem(val))
        except Exception as e:
            self.table_preview.clear()
            self.table_preview.setRowCount(0)
            self.table_preview.setColumnCount(1)
            self.table_preview.setHorizontalHeaderLabels(["解析错误"])
            self.table_preview.setItem(0, 0, QTableWidgetItem(str(e)))

    def accept_data(self):
        sep_key = self.combo_sep.currentText()
        sep = self.delimiter_map[sep_key]
        skiprows = self.spin_skip.value()
        x_col = self.spin_x_col.value()
        y_col = self.spin_y_col.value()
        try:
            usecols_list = list(set([x_col, y_col]))
            df = pd.read_csv(self.filepath, sep=sep, skiprows=skiprows, usecols=usecols_list, engine='python', header=None)
            for i in range(df.shape[1]):
                col_name = df.columns[i]
                clean_str = df[col_name].astype(str).str.replace(r'[,;]', '', regex=True)
                df[col_name] = pd.to_numeric(clean_str, errors='coerce')
            df.dropna(inplace=True)
            if df.empty:
                raise ValueError("未找到有效的数值数据。请检查分隔符设置，或跳过行数是否正确。")
            if x_col == y_col:
                self.final_x = df.iloc[:, 0].values
                self.final_y = df.iloc[:, 0].values
            else:
                if x_col < y_col:
                    self.final_x = df.iloc[:, 0].values
                    self.final_y = df.iloc[:, 1].values
                else:
                    self.final_x = df.iloc[:, 1].values
                    self.final_y = df.iloc[:, 0].values
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "数据加载错误", f"按当前设置加载数据失败:\n{str(e)}")

class GenericMplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()

class BaseAnalysisWidget(QWidget):
    """底层核心绘图与交互基类，封装图表共性逻辑、数据容器与UI基础骨架"""
    def __init__(self):
        super().__init__()
        self.plot_data_registry = {}
        self.custom_annotations = []
        self.is_annotation_mode = False
        
        self.main_layout = QVBoxLayout(self)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setMinimumWidth(360)
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        
        self.setup_shared_ui()
        
        self.left_layout.addStretch()
        self.left_scroll.setWidget(self.left_widget)
        self.main_splitter.addWidget(self.left_scroll)
        
        # 右侧画布区域
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas_scroll = QScrollArea()
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = GenericMplCanvas(self, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        self.canvas_scroll.setWidget(self.canvas_container)
        self.right_splitter.addWidget(self.canvas_scroll)
        
        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setSizes([360, 1040])
        
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def setup_shared_ui(self):
        # 1. 数据管理控制面板
        self.data_group = QGroupBox("数据载入与序列管理")
        data_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("导入文件")
        self.btn_clear = QPushButton("清除所有")
        self.btn_load.clicked.connect(self.load_files)
        self.btn_clear.clicked.connect(self.clear_plot)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_clear)
        data_layout.addLayout(btn_layout)
        
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["显示", "原始文件", "图例名称(双击修改)"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.file_table.itemChanged.connect(self.on_table_item_changed)
        data_layout.addWidget(self.file_table)
        self.data_group.setLayout(data_layout)
        self.left_layout.addWidget(self.data_group)
        
        # 2. 图表外观控制
        self.appearance_group = QGroupBox("图表基础外观")
        appearance_layout = QFormLayout()
        self.input_title = QLineEdit("Data Analysis Plot")
        self.input_xlabel = QLineEdit("X Axis")
        self.input_ylabel = QLineEdit("Y Axis")
        self.input_title.textChanged.connect(self.update_plot)
        self.input_xlabel.textChanged.connect(self.update_plot)
        self.input_ylabel.textChanged.connect(self.update_plot)
        appearance_layout.addRow("图表标题:", self.input_title)
        appearance_layout.addRow("X轴标签:", self.input_xlabel)
        appearance_layout.addRow("Y轴标签:", self.input_ylabel)
        self.appearance_group.setLayout(appearance_layout)
        self.left_layout.addWidget(self.appearance_group)
        
        # 3. 尺寸比例控制
        self.view_group = QGroupBox("视图布局控制")
        view_layout = QVBoxLayout()
        self.combo_ratio = QComboBox()
        self.combo_ratio.addItems(["自适应充满屏幕", "16:9 宽屏模式", "4:3 标准模式", "1:1 方形模式"])
        self.combo_ratio.currentIndexChanged.connect(self.apply_chart_ratio)
        view_layout.addWidget(self.combo_ratio)
        self.view_group.setLayout(view_layout)
        self.left_layout.addWidget(self.view_group)
        
        # 4. 全局交互工具
        self.interactive_group = QGroupBox("全局交互工具")
        interactive_layout = QVBoxLayout()
        self.btn_annotate = QPushButton("开启文本标注模式")
        self.btn_annotate.setCheckable(True)
        self.btn_annotate.clicked.connect(self.toggle_annotation_mode)
        self.btn_clear_custom = QPushButton("清除所有自定义标记")
        self.btn_clear_custom.clicked.connect(self.clear_custom_drawings)
        interactive_layout.addWidget(self.btn_annotate)
        interactive_layout.addWidget(self.btn_clear_custom)
        self.interactive_group.setLayout(interactive_layout)
        self.left_layout.addWidget(self.interactive_group)

    def load_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择数据文件", "", "Data Files (*.csv *.txt *.dat);;All Files (*)")
        if not file_paths:
            return
            
        self.file_table.blockSignals(True)
        for path in file_paths:
            if path in self.plot_data_registry:
                continue 
                
            dialog = DataImportDialog(path, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                filename = os.path.basename(path)
                self.plot_data_registry[path] = {
                    'filename': filename,
                    'x': dialog.final_x,
                    'y': dialog.final_y,
                    'original_y': dialog.final_y.copy(),
                    'auto_peaks_x': [],
                    'auto_peaks_y': [],
                    'auto_peak_areas': [],
                    'auto_peak_bounds': [],
                    'custom_areas': []
                }
                
                row_position = self.file_table.rowCount()
                self.file_table.insertRow(row_position)
                
                item_check = QTableWidgetItem()
                item_check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item_check.setCheckState(Qt.CheckState.Checked)
                item_check.setData(Qt.ItemDataRole.UserRole, path)
                
                item_filename = QTableWidgetItem(filename)
                item_filename.setFlags(item_filename.flags() & ~Qt.ItemFlag.ItemIsEditable)
                
                item_label = QTableWidgetItem(filename)
                
                self.file_table.setItem(row_position, 0, item_check)
                self.file_table.setItem(row_position, 1, item_filename)
                self.file_table.setItem(row_position, 2, item_label)
                
        self.file_table.blockSignals(False)
        self.update_plot()

    def on_table_item_changed(self, item):
        self.update_plot()

    def get_checked_paths(self):
        """返回当前表格中被勾选的文件路径列表"""
        checked_paths = []
        for row in range(self.file_table.rowCount()):
            chk_item = self.file_table.item(row, 0)
            if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
                checked_paths.append(chk_item.data(Qt.ItemDataRole.UserRole))
        return checked_paths

    def get_display_name(self, path):
        """获取特定路径对应的自定义图例名称"""
        for row in range(self.file_table.rowCount()):
            chk_item = self.file_table.item(row, 0)
            if chk_item and chk_item.data(Qt.ItemDataRole.UserRole) == path:
                return self.file_table.item(row, 2).text()
        return os.path.basename(path)

    def apply_chart_ratio(self):
        ratio_text = self.combo_ratio.currentText()
        if "自适应" in ratio_text:
            self.canvas_container.setMinimumSize(0, 0)
            self.canvas_container.setMaximumSize(16777215, 16777215)
            self.canvas_scroll.setWidgetResizable(True)
        elif "16:9" in ratio_text:
            self.canvas_scroll.setWidgetResizable(False)
            self.canvas_container.setFixedSize(1280, 720)
        elif "4:3" in ratio_text:
            self.canvas_scroll.setWidgetResizable(False)
            self.canvas_container.setFixedSize(1024, 768)
        elif "1:1" in ratio_text:
            self.canvas_scroll.setWidgetResizable(False)
            self.canvas_container.setFixedSize(800, 800)
        self.update_plot()

    def toggle_annotation_mode(self):
        self.is_annotation_mode = self.btn_annotate.isChecked()
        if self.is_annotation_mode:
            self.btn_annotate.setText("标注模式进行中...(点击图表)")
            self.btn_annotate.setStyleSheet("background-color: #f39c12; color: white;")
        else:
            self.btn_annotate.setText("开启文本标注模式")
            self.btn_annotate.setStyleSheet("")

    def on_canvas_click(self, event):
        if event.button != 1 or not self.is_annotation_mode or event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        text, ok = QInputDialog.getText(self, "添加标注", "请输入要标记在图表上的文本内容:")
        if ok and text:
            self.custom_annotations.append((x, y, text))
            self.btn_annotate.setChecked(False)
            self.toggle_annotation_mode()
            self.update_plot()

    def clear_custom_drawings(self):
        self.custom_annotations.clear()
        for data in self.plot_data_registry.values():
            data['custom_areas'] = []
        self.update_plot()

    def clear_plot(self):
        self.plot_data_registry.clear()
        self.file_table.setRowCount(0)
        self.custom_annotations.clear()
        self.canvas.ax.clear()
        self.canvas.ax.grid(False)
        self.canvas.draw()

    def apply_base_plot_settings(self):
        """执行通用的图表属性渲染指令"""
        if self.plot_data_registry:
            self.canvas.ax.legend(loc='upper right')
            self.canvas.ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas.ax.set_title(self.input_title.text())
        self.canvas.ax.set_xlabel(self.input_xlabel.text())
        self.canvas.ax.set_ylabel(self.input_ylabel.text())
        
        for (x, y, text) in self.custom_annotations:
            self.canvas.ax.annotate(text, xy=(x, y), xytext=(15, 15), textcoords='offset points',
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.8))
        self.canvas.fig.tight_layout()

    def update_plot(self, *args):
        """抽象方法，交由子类实现具体的渲染逻辑"""
        pass

# ==================== 模块 1: 多文件叠加绘图模块 ====================

class MultiPlotterWidget(BaseAnalysisWidget):
    def __init__(self):
        super().__init__()
        self.input_title.setText("Multi-file Data Overlay Plot")

    def update_plot(self, *args):
        self.canvas.ax.clear()
        checked_paths = self.get_checked_paths()
        
        for path in checked_paths:
            data = self.plot_data_registry[path]
            display_name = self.get_display_name(path)
            self.canvas.ax.plot(data['x'], data['y'], label=display_name, linewidth=1.5)
            
        self.apply_base_plot_settings()
        self.canvas.draw()

# ==================== 模块 2: 单/多谱图并发高级分析模块 ====================

class SpectrumAnalyzerWidget(BaseAnalysisWidget):
    def __init__(self):
        super().__init__()
        self.input_title.setText("Advanced Spectrum Analysis")
        self.add_spectrum_tools()

    def add_spectrum_tools(self):
        # 注入分析模块专属的控制组（插入在交互工具上方）
        insert_idx = self.left_layout.indexOf(self.interactive_group)
        
        self.preprocess_group = QGroupBox("选中序列批处理预处理")
        preprocess_layout = QVBoxLayout()
        self.btn_smooth = QPushButton("平滑降噪 (对勾选序列)")
        self.btn_baseline = QPushButton("自动扣基线 (对勾选序列)")
        self.btn_norm = QPushButton("归一化0-1 (对勾选序列)")
        self.btn_reset = QPushButton("重置数据 (对勾选序列)")
        self.btn_smooth.clicked.connect(self.apply_smoothing_sg)
        self.btn_baseline.clicked.connect(self.apply_baseline_correction_arpls)
        self.btn_norm.clicked.connect(self.apply_normalization)
        self.btn_reset.clicked.connect(self.reset_data)
        preprocess_layout.addWidget(self.btn_smooth)
        preprocess_layout.addWidget(self.btn_baseline)
        preprocess_layout.addWidget(self.btn_norm)
        preprocess_layout.addWidget(self.btn_reset)
        self.preprocess_group.setLayout(preprocess_layout)
        self.left_layout.insertWidget(insert_idx, self.preprocess_group)
        
        self.peak_group = QGroupBox("自适应并发寻峰参数")
        peak_layout = QFormLayout()
        self.spin_prominence = QDoubleSpinBox()
        self.spin_prominence.setRange(0.001, 100000)
        self.spin_prominence.setValue(15.0)
        self.spin_prominence.setSingleStep(0.05)
        peak_layout.addRow("Prominence(相对阈值):", self.spin_prominence)
        
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.0, 100000)
        self.spin_height.setValue(0.02)
        self.spin_height.setSingleStep(0.01)
        peak_layout.addRow("Min Height(最小高):", self.spin_height)
        
        self.btn_find_peaks = QPushButton("对勾选序列执行寻峰")
        self.btn_find_peaks.setStyleSheet("background-color: #2c3e50; color: white; padding: 5px;")
        self.btn_find_peaks.clicked.connect(self.execute_auto_peaks)
        peak_layout.addRow(self.btn_find_peaks)
        
        self.btn_export = QPushButton("导出分析结果(CSV)")
        self.btn_export.clicked.connect(self.export_results)
        peak_layout.addRow(self.btn_export)
        
        self.peak_group.setLayout(peak_layout)
        self.left_layout.insertWidget(insert_idx + 1, self.peak_group)

        # 添加底部日志窗口
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.right_splitter.addWidget(self.log_text)
        self.right_splitter.setSizes([850, 150])

        # 注册框选积分事件
        self.span = SpanSelector(
            self.canvas.ax, 
            self.on_span_select, 
            'horizontal', 
            useblit=True,
            props=dict(alpha=0.3, facecolor='green'),
            interactive=False
        )

    def log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def apply_smoothing_sg(self):
        paths = self.get_checked_paths()
        if not paths:
            self.log("错误: 当前没有勾选任何数据。")
            return
        for path in paths:
            data = self.plot_data_registry[path]
            window_length = min(51, len(data['y']) if len(data['y']) % 2 != 0 else len(data['y']) - 1)
            if window_length > 3:
                data['y'] = savgol_filter(data['y'], window_length=window_length, polyorder=3)
                self.clear_analysis_data(path)
                self.log(f"对 {data['filename']} 应用了 SG 平滑降噪。")
        self.update_plot()

    def apply_baseline_correction_arpls(self):
        paths = self.get_checked_paths()
        for path in paths:
            data = self.plot_data_registry[path]
            try:
                baseline = baseline_arpls(data['x'], data['y'], lam=1e6, niter=15)
                data['y'] = data['y'] - baseline
                self.clear_analysis_data(path)
                self.log(f"对 {data['filename']} 完成了基线拉平。")
            except Exception as e:
                self.log(f"对 {data['filename']} 基线拟合失败: {str(e)}")
        self.update_plot()

    def apply_normalization(self):
        paths = self.get_checked_paths()
        for path in paths:
            data = self.plot_data_registry[path]
            y_min = np.min(data['y'])
            y_max = np.max(data['y'])
            if y_max - y_min != 0:
                data['y'] = (data['y'] - y_min) / (y_max - y_min)
                self.clear_analysis_data(path)
                self.log(f"对 {data['filename']} 应用了归一化。")
        self.spin_prominence.setValue(0.05)
        self.spin_height.setValue(0.02)
        self.update_plot()

    def reset_data(self):
        paths = self.get_checked_paths()
        for path in paths:
            data = self.plot_data_registry[path]
            data['y'] = data['original_y'].copy()
            self.clear_analysis_data(path)
            self.log(f"{data['filename']} 数据已重置。")
        self.update_plot()

    def clear_analysis_data(self, path):
        data = self.plot_data_registry[path]
        data['auto_peaks_x'] = []
        data['auto_peaks_y'] = []
        data['auto_peak_areas'] = []
        data['auto_peak_bounds'] = []
        data['custom_areas'] = []

    def execute_auto_peaks(self):
        paths = self.get_checked_paths()
        if not paths:
            self.log("错误: 没有勾选数据执行寻峰。")
            return
            
        prominence = self.spin_prominence.value()
        height = self.spin_height.value()
        
        for path in paths:
            data = self.plot_data_registry[path]
            self.log(f"正在对 {data['filename']} 执行并发寻峰分析...")

            adaptive_distance = max(1, len(data['y']) // 1000)
            adaptive_width = max(1.0, len(data['y']) // 2000)
            
            peaks_indices, properties = find_peaks(data['y'], prominence=prominence, height=height, distance=adaptive_distance, width=adaptive_width)
            
            if len(peaks_indices) == 0:
                self.log(f"{data['filename']} 未找到满足阈值的谱峰。")
                continue
                
            data['auto_peaks_x'] = data['x'][peaks_indices]
            data['auto_peaks_y'] = data['y'][peaks_indices]
            
            valleys = []
            for i in range(len(peaks_indices) - 1):
                p1 = peaks_indices[i]
                p2 = peaks_indices[i+1]
                valley_idx = p1 + np.argmin(data['y'][p1:p2])
                valleys.append(valley_idx)

            left_bases = properties["left_bases"]
            right_bases = properties["right_bases"]

            data['auto_peak_areas'] = []
            data['auto_peak_bounds'] = []
            
            self.log(f"序列 {data['filename']} 成功识别 {len(peaks_indices)} 个峰:")
            
            for i in range(len(peaks_indices)):
                nat_left = left_bases[i]
                nat_right = right_bases[i]
                
                if i > 0:
                    left_bound = max(nat_left, valleys[i-1])
                else:
                    left_bound = nat_left
                    
                if i < len(peaks_indices) - 1:
                    right_bound = min(nat_right, valleys[i])
                else:
                    right_bound = nat_right

                data['auto_peak_bounds'].append((left_bound, right_bound))
                
                x_peak = data['x'][left_bound:right_bound+1]
                y_peak = data['y'][left_bound:right_bound+1]
                
                if len(x_peak) > 1:
                    base_val = np.min(data['y'])
                    local_baseline = np.full(len(x_peak), base_val)
                    pure_y = y_peak - local_baseline
                    pure_y = np.clip(pure_y, 0, None)
                    try:
                        area = abs(simpson(pure_y, x=x_peak))
                    except TypeError:
                        area = abs(simpson(y=pure_y, x=x_peak))
                    except Exception:
                        area = np.sum(pure_y) * (x_peak[-1] - x_peak[0]) / len(x_peak)
                else:
                    area = 0.0
                    
                data['auto_peak_areas'].append(area)
                self.log(f"  峰{i+1}: X={data['auto_peaks_x'][i]:.2f}, 面积={area:.2f}")
                
        self.update_plot()

    def on_span_select(self, xmin, xmax):
        paths = self.get_checked_paths()
        if not paths:
            return
        if xmin == xmax:
            return
        if xmin > xmax:
            xmin, xmax = xmax, xmin
            
        for path in paths:
            data = self.plot_data_registry[path]
            mask = (data['x'] >= xmin) & (data['x'] <= xmax)
            if not np.any(mask):
                continue
                
            x_sel = data['x'][mask]
            y_sel = data['y'][mask]
            
            if len(x_sel) > 1:
                base_val = np.min(data['y'])
                local_baseline = np.full(len(x_sel), base_val)
                pure_y = y_sel - local_baseline
                pure_y = np.clip(pure_y, 0, None)
                
                try:
                    area = abs(simpson(pure_y, x=x_sel))
                except TypeError:
                    area = abs(simpson(y=pure_y, x=x_sel))
                except Exception:
                    area = np.sum(pure_y) * (x_sel[-1] - x_sel[0]) / len(x_sel)
                
                data['custom_areas'].append((xmin, xmax, local_baseline, area, x_sel, y_sel))
                self.log(f"[{data['filename']} 自定义积分] X: {xmin:.2f}~{xmax:.2f} | 面积: {area:.2f}")
                
        self.update_plot()

    def update_plot(self, *args):
        self.canvas.ax.clear()
        checked_paths = self.get_checked_paths()
        
        auto_colors = ['#e74c3c', '#3498db', '#9b59b6']
        
        for path_idx, path in enumerate(checked_paths):
            data = self.plot_data_registry[path]
            display_name = self.get_display_name(path)
            
            line_plot = self.canvas.ax.plot(data['x'], data['y'], label=display_name, linewidth=1.5)
            base_color = line_plot[0].get_color()
            
            if len(data['auto_peaks_x']) > 0:
                self.canvas.ax.plot(data['auto_peaks_x'], data['auto_peaks_y'], "x", color=base_color, markersize=8, markeredgewidth=2)
                for i, (left_idx, right_idx) in enumerate(data['auto_peak_bounds']):
                    x_peak = data['x'][left_idx:right_idx+1]
                    y_peak = data['y'][left_idx:right_idx+1]
                    base_val = np.min(data['y'])
                    local_baseline = np.full(len(x_peak), base_val)
                    c = auto_colors[i % len(auto_colors)]
                    y_top = np.maximum(y_peak, local_baseline)
                    self.canvas.ax.fill_between(x_peak, y_top, local_baseline, color=c, alpha=0.3)
                    self.canvas.ax.plot([x_peak[0], x_peak[-1]], [base_val, base_val], color=c, linestyle='--', alpha=0.8)

            for i, (xmin, xmax, local_baseline, area, x_sel, y_sel) in enumerate(data['custom_areas']):
                y_top = np.maximum(y_sel, local_baseline)
                self.canvas.ax.fill_between(x_sel, y_top, local_baseline, color=base_color, alpha=0.4)
                base_val = local_baseline[0]
                self.canvas.ax.plot([x_sel[0], x_sel[-1]], [base_val, base_val], color=base_color, linestyle='--', alpha=0.8)
                
                mid_x = (xmin + xmax) / 2
                max_y_sel = np.max(y_sel)
                y_range = np.max(data['y']) - np.min(data['y']) if np.max(data['y']) != np.min(data['y']) else 1
                text_offset = y_range * 0.02
                self.canvas.ax.text(mid_x, max_y_sel + text_offset, f'{area:.2f}', color=base_color, fontweight='bold', ha='center', va='bottom', fontsize=8)

        self.apply_base_plot_settings()
        self.canvas.draw()

    def export_results(self):
        paths = self.get_checked_paths()
        has_data = any(len(self.plot_data_registry[p]['auto_peak_areas']) > 0 or len(self.plot_data_registry[p]['custom_areas']) > 0 for p in paths)
        
        if not has_data:
            QMessageBox.warning(self, "无数据", "当前勾选的序列没有分析结果可以导出。")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "保存分析结果", "", "CSV Files (*.csv)")
        if not filename:
            return
            
        try:
            records = []
            for path in paths:
                data = self.plot_data_registry[path]
                src_name = data['filename']
                
                for i in range(len(data['auto_peaks_x'])):
                    records.append({
                        "来源文件": src_name,
                        "类型": "自动寻峰",
                        "峰编号": i + 1,
                        "X坐标_峰位置": data['auto_peaks_x'][i],
                        "Y强度_峰高": data['auto_peaks_y'][i],
                        "峰面积": data['auto_peak_areas'][i],
                        "起始X边界": data['x'][data['auto_peak_bounds'][i][0]],
                        "结束X边界": data['x'][data['auto_peak_bounds'][i][1]]
                    })
                    
                for i, (xmin, xmax, local_baseline, area, x_sel, y_sel) in enumerate(data['custom_areas']):
                    records.append({
                        "来源文件": src_name,
                        "类型": "自定义积分",
                        "峰编号": i + 1,
                        "X坐标_中心位置": (xmin + xmax) / 2,
                        "Y强度_最高点": np.max(y_sel),
                        "峰面积": area,
                        "起始X边界": xmin,
                        "结束X边界": xmax
                    })
                    
            df_export = pd.DataFrame(records)
            df_export.to_csv(filename, index=False, encoding='utf-8-sig') 
            self.log(f"成功导出分析结果至: {filename}")
            QMessageBox.information(self, "导出成功", f"结果已成功保存。")
        except Exception as e:
            self.log(f"导出失败: {str(e)}")
            QMessageBox.critical(self, "导出错误", f"保存文件时发生错误:\n{str(e)}")

# ==================== 主系统集成 ====================

class IntegratedPlottingSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("综合实验数据与谱图分析系统")
        self.resize(1400, 850)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.multi_plotter_tab = MultiPlotterWidget()
        self.spectrum_analyzer_tab = SpectrumAnalyzerWidget()
        
        self.tabs.addTab(self.multi_plotter_tab, "多文件叠加绘图模块")
        self.tabs.addTab(self.spectrum_analyzer_tab, "多序列并发分析模块")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntegratedPlottingSystem()
    window.show()
    sys.exit(app.exec())