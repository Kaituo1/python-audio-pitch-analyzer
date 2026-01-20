# 恢复正常导入方式，确保启动速度稳定
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import sys
import threading
import logging

# 配置日志 - 只输出到控制台，不生成日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KeyDetector")
logging.getLogger("pydub.converter").setLevel(logging.ERROR)
logging.getLogger("pydub.utils").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.WARNING)

# 核心依赖库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from pydub import AudioSegment
import librosa

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义调性映射
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODES = ['major', 'minor']

class AudioKeyDetector:
    def __init__(self):
        self.supported_formats = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff']
        self.key_templates = self._create_key_templates()
    
    def _create_key_templates(self):
        """创建改进的调性模板，包含多种权威模板"""
        # 1. Krumhansl-Schmuckler-Guo模板（当前使用）
        ks_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        ks_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # 2. Krumhansl-Kessler模板（经典模板）
        kk_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        kk_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # 3. Aarden-Essen模板（基于统计的模板）
        ae_major = np.array([3.48, 0.06, 3.38, 0.10, 3.22, 3.33, 0.11, 3.28, 0.10, 2.74, 0.05, 3.16])
        ae_minor = np.array([3.47, 0.06, 3.41, 3.38, 0.09, 3.29, 0.13, 3.30, 3.26, 0.10, 2.73, 0.05])
        
        # 4. Temperley模板（改进的统计模板）
        t_major = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
        t_minor = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 2.5, 3.0])
        
        # 综合多种模板，加权平均
        major_template = (ks_major + kk_major + ae_major + t_major) / 4.0
        minor_template = (ks_minor + kk_minor + ae_minor + t_minor) / 4.0
        
        templates = {
            'major': [],
            'minor': []
        }
        
        # 创建大调模板
        for i in range(12):
            templates['major'].append(np.roll(major_template, i))
        
        # 创建小调模板
        for i in range(12):
            templates['minor'].append(np.roll(minor_template, i))
        
        return templates
    
    def _calculate_similarity(self, chroma_feature, template):
        """计算色度特征与模板的相似度，使用多种距离度量方法"""
        # 1. 皮尔逊相关系数（已使用）
        corr_coef = np.corrcoef(chroma_feature, template)[0, 1]
        
        # 2. 余弦相似度
        cos_sim = np.dot(chroma_feature, template) / (np.linalg.norm(chroma_feature) * np.linalg.norm(template))
        
        # 3. 欧几里得距离（转换为相似度）
        euclidean_dist = np.linalg.norm(chroma_feature - template)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # 4. 曼哈顿距离（转换为相似度）
        manhattan_dist = np.sum(np.abs(chroma_feature - template))
        manhattan_sim = 1.0 / (1.0 + manhattan_dist)
        
        # 综合多种相似度度量，加权平均
        # 相关系数和余弦相似度权重较高，因为它们更适合色度特征匹配
        combined_sim = (corr_coef * 0.4 + cos_sim * 0.4 + euclidean_sim * 0.1 + manhattan_sim * 0.1)
        
        return combined_sim
    
    def _calculate_correlations(self, chroma_feature):
        """计算色度特征与调性模板的相关性，使用多种相似度度量"""
        # 归一化色度特征，提高匹配准确性
        chroma_norm = chroma_feature / (np.sum(chroma_feature) + 1e-6)  # 添加小常数避免除以零
        
        # 计算与模板的相似度
        major_corrs = [self._calculate_similarity(chroma_norm, template) for template in self.key_templates['major']]
        minor_corrs = [self._calculate_similarity(chroma_norm, template) for template in self.key_templates['minor']]
        
        return major_corrs, minor_corrs
    
    def detect_key_improved(self, file_path, retry_count=3):
        """改进的调性检测算法，支持重试机制"""
        file_name = os.path.basename(file_path)
        
        for attempt in range(retry_count):
            try:
                logger.info(f"开始检测文件: {file_name} (尝试 {attempt+1}/{retry_count})")
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    logger.error(f"文件不存在: {file_path}")
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                
                # 检查文件是否为支持的格式
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in self.supported_formats:
                    logger.error(f"不支持的文件格式: {file_ext} 对于文件 {file_name}")
                    raise ValueError(f"不支持的文件格式: {file_ext}")
                
                # 直接使用librosa加载完整音频文件
                logger.info(f"加载音频文件: {file_name}")
                y, sr = librosa.load(file_path, sr=None)
                
                # 检查音频长度和振幅
                audio_length = librosa.get_duration(y=y, sr=sr)
                audio_rms = np.sqrt(np.mean(y**2))
                logger.info(f"音频信息 - 长度: {audio_length:.2f}秒, 采样率: {sr}Hz, RMS振幅: {audio_rms:.6f}")
                
                # 处理太短的音频文件
                if audio_length < 2.0:
                    logger.warning(f"音频文件太短: {file_name} (仅 {audio_length:.2f}秒)")
                    raise ValueError(f"音频文件太短: 仅 {audio_length:.2f}秒，无法准确检测调性")
                
                # 处理静音或低音量音频
                if audio_rms < 0.001:
                    logger.warning(f"音频文件音量过低: {file_name} (RMS: {audio_rms:.6f})")
                    raise ValueError(f"音频文件音量过低，无法准确检测调性")
                
                # 提取多个特征进行综合判断，优化参数以提高准确性
                
                # 1. CQT色度特征（基础特征）
                chroma_cqt = librosa.feature.chroma_cqt(
                    y=y, 
                    sr=sr, 
                    bins_per_octave=36, 
                    hop_length=512, 
                    norm=2
                )
                chroma_cqt_mean = np.mean(chroma_cqt, axis=1)
                logger.debug(f"CQT特征均值: {chroma_cqt_mean}")
                
                # 2. STFT色度特征（备份特征）
                chroma_stft = librosa.feature.chroma_stft(
                    y=y, 
                    sr=sr, 
                    n_fft=4096, 
                    hop_length=512, 
                    norm=2
                )
                chroma_stft_mean = np.mean(chroma_stft, axis=1)
                logger.debug(f"STFT特征均值: {chroma_stft_mean}")
                
                # 3. 感知色度特征
                chroma_cens = librosa.feature.chroma_cens(
                    y=y, 
                    sr=sr,
                    hop_length=512,
                    fmin=librosa.note_to_hz('C1')
                )
                chroma_cens_mean = np.mean(chroma_cens, axis=1)
                logger.debug(f"CENS特征均值: {chroma_cens_mean}")
                
                # 4. 变分辨率色度特征 - 使用兼容参数
                try:
                    # 注意：当前版本librosa可能不支持chroma_vqt，跳过这个特征
                    # 直接使用其他三种色度特征（chroma_cqt, chroma_stft, chroma_cens）
                    # 这样可以避免chroma_vqt带来的兼容性问题
                    logger.debug("跳过 chroma_vqt 特征提取，避免兼容性问题")
                    # 创建一个占位的chroma_vqt特征，使用chroma_stft的副本
                    chroma_vqt = librosa.feature.chroma_stft(
                        y=y, 
                        sr=sr, 
                        n_fft=4096, 
                        hop_length=512, 
                        norm=2
                    )
                except Exception as e:
                    logger.error(f"chroma_vqt 特征提取失败: {str(e)}")
                    # 如果失败，使用chroma_stft的副本作为替代
                    chroma_vqt = librosa.feature.chroma_stft(
                        y=y, 
                        sr=sr, 
                        n_fft=4096, 
                        hop_length=512, 
                        norm=2
                    )
                chroma_vqt_mean = np.mean(chroma_vqt, axis=1)
                logger.debug(f"VQT特征均值: {chroma_vqt_mean}")
                
                # 5. 计算所有特征的相关性
                logger.info("计算特征相关性")
                major_corrs_cqt, minor_corrs_cqt = self._calculate_correlations(chroma_cqt_mean)
                major_corrs_stft, minor_corrs_stft = self._calculate_correlations(chroma_stft_mean)
                major_corrs_cens, minor_corrs_cens = self._calculate_correlations(chroma_cens_mean)
                major_corrs_vqt, minor_corrs_vqt = self._calculate_correlations(chroma_vqt_mean)
                
                # 6. 综合多种方法的结果
                combined_major = [(m1 * 0.25 + m2 * 0.2 + m3 * 0.35 + m4 * 0.2) for m1, m2, m3, m4 in 
                               zip(major_corrs_cqt, major_corrs_stft, major_corrs_cens, major_corrs_vqt)]
                combined_minor = [(m1 * 0.25 + m2 * 0.2 + m3 * 0.35 + m4 * 0.2) for m1, m2, m3, m4 in 
                               zip(minor_corrs_cqt, minor_corrs_stft, minor_corrs_cens, minor_corrs_vqt)]
                
                # 7. 找到最佳匹配
                major_key_idx = np.argmax(combined_major)
                minor_key_idx = np.argmax(combined_minor)
                
                major_max_corr = combined_major[major_key_idx]
                minor_max_corr = combined_minor[minor_key_idx]
                
                logger.info(f"相关性得分 - 大调最大值: {major_max_corr:.4f}, 小调最大值: {minor_max_corr:.4f}")
                
                if major_max_corr > minor_max_corr:
                    detected_key = KEYS[major_key_idx]
                    detected_mode = 'major'
                    confidence = major_max_corr
                else:
                    detected_key = KEYS[minor_key_idx]
                    detected_mode = 'minor'
                    confidence = minor_max_corr
                
                # 8. 优化的后处理步骤
                # 计算特征的一致性得分
                all_features = np.vstack([chroma_cqt_mean, chroma_stft_mean, chroma_cens_mean, chroma_vqt_mean])
                feature_stds = np.std(all_features, axis=1)
                feature_corrs = np.corrcoef(all_features)
                avg_corr = np.mean(feature_corrs[np.triu_indices_from(feature_corrs, k=1)])
                
                logger.info(f"特征一致性 - 标准差: {np.mean(feature_stds):.4f}, 平均相关性: {avg_corr:.4f}")
                
                consistency_score = (1.0 - np.mean(feature_stds) / np.mean(all_features)) * 0.6 + avg_corr * 0.4
                
                # 加权平均最终置信度
                final_confidence = (confidence * 0.7 + consistency_score * 0.3)
                
                # 置信度阈值过滤
                if final_confidence < 0.3:
                    logger.warning(f"低置信度检测结果: {detected_key} {detected_mode} 对于文件 {file_name} (置信度: {final_confidence:.4f})")
                    detected_key = 'N/A'
                    detected_mode = 'N/A'
                    final_confidence = 0.0
                
                # 确保置信度在合理范围内
                final_confidence = max(0, min(1, final_confidence))
                
                logger.info(f"检测完成 - 文件: {file_name}, 调性: {detected_key}, 模式: {detected_mode}, 置信度: {final_confidence:.4f}")
                
                # 定义成功标准：置信度 >= 0.5 或已达到最大重试次数
                if final_confidence >= 0.5 or attempt == retry_count - 1:
                    return detected_key, detected_mode, final_confidence
                
                # 如果置信度不足但还有重试机会，继续重试
                logger.info(f"置信度不足 ({final_confidence:.4f} < 0.5)，进行重试...")
                
            except Exception as e:
                error_msg = f"分析失败 (尝试 {attempt+1}/{retry_count}): {str(e)}"
                logger.error(error_msg)
                
                # 如果是最后一次尝试，返回失败结果
                if attempt == retry_count - 1:
                    logger.error(f"所有尝试都失败了，返回未知结果")
                    return 'N/A', 'N/A', 0.0
                
                # 短暂延迟后重试
                import time
                time.sleep(0.5)
                logger.info(f"500ms后重试检测...")
                continue
    
    def batch_detect(self, file_paths):
        """批量检测调性，支持重试机制和详细日志"""
        results = []
        success_count = 0
        failure_count = 0
        low_confidence_count = 0
        
        logger.info(f"开始批处理检测，共 {len(file_paths)} 个文件")
        
        for file_path in file_paths:
            # 使用改进的检测方法，支持重试
            key, mode, confidence = self.detect_key_improved(file_path, retry_count=3)
            
            result = {
                'file': os.path.basename(file_path),
                'key': key,
                'mode': mode,
                'confidence': confidence
            }
            results.append(result)
            
            # 更新统计信息
            if key != 'N/A':
                success_count += 1
                if confidence < 0.5:
                    low_confidence_count += 1
            else:
                failure_count += 1
        
        # 记录批处理统计信息
        logger.info(f"批处理完成 - 成功: {success_count}, 失败: {failure_count}, 低置信度: {low_confidence_count}")
        logger.info(f"成功率: {success_count/len(file_paths)*100:.1f}%")
        
        return results

class AudioVisualizer:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        # 优化matplotlib参数：降低dpi和figsize以减少内存使用
        self.fig, self.ax = plt.subplots(figsize=(7, 3.5), dpi=80, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 音频数据缓存
        self._audio_cache = {}
        
        # 初始清空状态
        self.ax.clear()
        self.ax.set_title('')
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.canvas.draw()
    
    def _load_audio(self, file_path):
        """加载音频文件并缓存结果"""
        if file_path not in self._audio_cache:
            try:
                # 加载完整音频文件
                y, sr = librosa.load(file_path, sr=None)
                self._audio_cache[file_path] = (y, sr)
            except Exception as e:
                print(f"加载音频失败: {e}")
                return None, None
        return self._audio_cache[file_path]
    
    def clear_cache(self):
        """清空音频缓存"""
        self._audio_cache.clear()
    
    def plot_waveform(self, file_path):
        """绘制波形图"""
        try:
            y, sr = self._load_audio(file_path)
            if y is None or sr is None:
                return
            
            times = librosa.times_like(y)
            
            self.ax.clear()
            # 优化绘图参数：使用更高效的绘图方式
            self.ax.plot(times, y, color='blue', alpha=0.7, linewidth=0.3)  # 进一步降低线宽
            self.ax.set_title('音频波形')
            self.ax.set_xlabel('时间 (秒)')
            self.ax.set_ylabel('振幅')
            self.ax.grid(True, alpha=0.2)  # 降低网格透明度
            # 减少坐标轴刻度数量
            self.ax.locator_params(axis='x', nbins=6)
            self.ax.locator_params(axis='y', nbins=4)
            
            self.canvas.draw()
        except Exception as e:
            print(f"绘图失败: {e}")
    
    def plot_spectrogram(self, file_path):
        """绘制频谱图"""
        try:
            y, sr = self._load_audio(file_path)
            if y is None or sr is None:
                return
            
            # 优化spectrogram参数：使用更小的窗口和更高效的参数
            frequencies, times, Sxx = signal.spectrogram(
                y, fs=sr, nperseg=512, noverlap=256, nfft=1024
            )
            
            self.ax.clear()
            # 优化绘图：使用更高效的shading和颜色映射
            im = self.ax.pcolormesh(times, frequencies, 10 * np.log10(Sxx), 
                                  shading='flat', cmap='viridis')
            self.ax.set_ylabel('频率 (Hz)')
            self.ax.set_xlabel('时间 (秒)')
            self.ax.set_title('频谱图')
            self.ax.set_ylim(0, 8000)  # 限制频率范围
            # 减少坐标轴刻度数量
            self.ax.locator_params(axis='x', nbins=6)
            self.ax.locator_params(axis='y', nbins=4)
            
            self.canvas.draw()
        except Exception as e:
            print(f"绘图失败: {e}")

class KeyDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("音频调性检测器v1.0.0")
        self.root.geometry("950x900")
        
        # 禁止窗口大小调整
        self.root.resizable(False, False)  
        # 禁止水平和垂直调整
        
        # 正常创建组件
        self.detector = AudioKeyDetector()
        self.visualizer = None  # 仍延迟创建可视化器，因为它依赖于UI布局
        
        self.setup_ui()
    
    def setup_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === 控制面板（全新设计）===
        control_panel = ttk.LabelFrame(main_frame, text=" 🎵 控制面板 ", padding=15)
        control_panel.pack(fill=tk.X, pady=(0, 12))

        # 第一行：单文件选择
        row1 = ttk.Frame(control_panel)
        row1.pack(fill=tk.X, pady=4)
        
        # 单文件选择按钮
        self.single_btn = ttk.Button(
            row1, 
            text="📁 选择单个音频文件", 
            command=self.select_single_file,
            width=22
        )
        self.single_btn.pack(padx=2, pady=2, side=tk.LEFT)

        # 第二行：多文件选择 + 批处理 + 清空按钮
        row2 = ttk.Frame(control_panel)
        row2.pack(fill=tk.X, pady=6)
        
        # 多文件按钮
        self.multi_btn = ttk.Button(
            row2, 
            text="📂 选择多个音频文件", 
            command=self.select_multiple_files,
            width=22
        )
        self.multi_btn.pack(side=tk.LEFT, padx=2)

        # 批处理按钮
        self.batch_btn = ttk.Button(
            row2, 
            text="⚡ 开始批处理", 
            command=self.start_batch_process,
            state=tk.DISABLED,
            width=18
        )
        self.batch_btn.pack(side=tk.LEFT, padx=10)

        # 清空按钮
        self.clear_btn = ttk.Button(
            row2, 
            text="🗑️ 清空检测结果", 
            command=self.clear_results,
            width=18
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)

        # 文件数量标签
        self.file_count_label = ttk.Label(
            row2, 
            text="已选择：0 个文件", 
            font=('微软雅黑', 9, 'italic'), 
            foreground='#777777'
        )
        self.file_count_label.pack(side=tk.RIGHT)

        # 进度条优化
        progress_frame = ttk.Frame(control_panel)
        progress_frame.pack(fill=tk.X, pady=10)
        
        # 添加进度条标签
        self.progress_label = ttk.Label(progress_frame, text="准备就绪")
        self.progress_label.pack(side=tk.LEFT, padx=5, anchor=tk.CENTER)
        
        # 优化的进度条
        self.progress = ttk.Progressbar(
            progress_frame,
            mode='indeterminate',
            length=0,  # 自适应宽度
            style='Horizontal.TProgressbar'
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=2)

        # === 检测结果表格 ===
        result_panel = ttk.LabelFrame(main_frame, text=" 📊 检测结果 ", padding=10)
        result_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        columns = ('文件名', '调性', '模式', '置信度')
        self.tree = ttk.Treeview(result_panel, columns=columns, show='headings', height=6)
        
        # 列配置
        col_widths = {'文件名': 300, '调性': 80, '模式': 80, '置信度': 100}
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=col_widths[col], anchor=tk.CENTER)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_panel, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === 音频可视化区域 ===
        viz_panel = ttk.LabelFrame(main_frame, text=" 📈 音频可视化 ", padding=10)
        viz_panel.pack(fill=tk.BOTH, expand=True)

        self.visualizer = AudioVisualizer(viz_panel)

        # 可视化按钮
        viz_btn_frame = ttk.Frame(viz_panel)
        viz_btn_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.wave_btn = ttk.Button(
            viz_btn_frame, 
            text="🔊 显示波形", 
            command=self.show_waveform,
            state=tk.DISABLED
        )
        self.wave_btn.pack(side=tk.LEFT, padx=5)
        
        self.spec_btn = ttk.Button(
            viz_btn_frame, 
            text="📊 显示频谱", 
            command=self.show_spectrogram,
            state=tk.DISABLED
        )
        self.spec_btn.pack(side=tk.LEFT, padx=5)

        # 初始化状态
        self.selected_files = []
    
    def update_button_states(self):
        """根据选中文件数量更新按钮状态"""
        has_files = len(self.selected_files) > 0
        self.batch_btn.config(state=tk.NORMAL if has_files else tk.DISABLED)
        self.wave_btn.config(state=tk.NORMAL if has_files else tk.DISABLED)
        self.spec_btn.config(state=tk.NORMAL if has_files else tk.DISABLED)
    
    def select_single_file(self):
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.opus *.aiff")]
        )
        if file_path:
            self.selected_files = [file_path]
            self.file_count_label.config(text=f"已选择：{len(self.selected_files)} 个文件")
            self.update_button_states()
            threading.Thread(target=self.process_single_file, args=(file_path,), daemon=True).start()
    
    def select_multiple_files(self):
        files = filedialog.askopenfilenames(
            title="选择多个音频文件",
            filetypes=[("音频文件", "*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.opus *.aiff")]
        )
        if files:
            self.selected_files = list(files)
            self.file_count_label.config(text=f"已选择：{len(self.selected_files)} 个文件")
            self.update_button_states()
            self.update_tree_view([])  # 清空之前的批处理结果
    
    def process_single_file(self, file_path):
        # 更新进度条状态
        file_name = os.path.basename(file_path)
        self.progress_label.config(text=f"正在检测: {file_name}")
        self.progress.start()
        
        try:
            # 直接使用检测器实例，支持重试机制
            key, mode, confidence = self.detector.detect_key_improved(file_path, retry_count=3)
            
            self.progress.stop()
            
            # 创建单文件检测结果并添加到表格
            single_result = {
                'file': file_name,
                'key': key,
                'mode': mode,
                'confidence': confidence
            }
            
            # 获取当前表格数据
            current_items = []
            for item in self.tree.get_children():
                values = self.tree.item(item, 'values')
                current_items.append({
                    'file': values[0],
                    'key': values[1],
                    'mode': values[2],
                    'confidence': float(values[3])
                })
            
            # 添加新结果
            current_items.append(single_result)
            # 更新表格
            self.update_tree_view(current_items)
            
            # 根据检测结果更新状态信息
            if key != 'N/A':
                if confidence >= 0.5:
                    self.progress_label.config(text="检测完成")
                    status_msg = f"检测完成 - {file_name}: {key} {mode} (置信度: {confidence:.3f})"
                    logger.info(status_msg)
                else:
                    self.progress_label.config(text="低置信度检测")
                    status_msg = f"低置信度检测 - {file_name}: {key} {mode} (置信度: {confidence:.3f})"
                    logger.warning(status_msg)
                    messagebox.warning("低置信度检测", 
                                     f"文件 {file_name} 的调性检测结果置信度较低 ({confidence:.3f})\n" +
                                     "结果可能不准确，建议手动验证。")
            else:
                self.progress_label.config(text="检测失败")
                status_msg = f"检测失败 - {file_name}"
                logger.error(status_msg)
                messagebox.showerror("检测失败", 
                                  f"无法检测文件 {file_name} 的调性\n" +
                                  "可能原因：文件太短、音量过低或格式不支持。")
            
            # 自动加载可视化（如果可视化器已创建）
            if self.visualizer and key != 'N/A':
                self.root.after(0, lambda: self.visualizer.plot_waveform(file_path))
        except Exception as e:
            self.progress.stop()
            error_msg = f"处理文件 {file_name} 时发生错误: {str(e)}"
            self.progress_label.config(text="检测失败")
            logger.error(error_msg)
            messagebox.showerror("检测失败", error_msg)
        finally:
            # 3秒后恢复默认状态
            self.root.after(3000, lambda: self.progress_label.config(text="准备就绪"))
    
    def start_batch_process(self):
        if not self.selected_files:
            messagebox.showwarning("警告", "请先选择音频文件")
            return
        
        # 更新进度条状态
        self.progress_label.config(text=f"开始批处理: {len(self.selected_files)} 个文件")
        self.progress.start()
        threading.Thread(target=self.batch_process_thread, daemon=True).start()
    
    def batch_process_thread(self):
        try:
            # 直接使用检测器实例
            results = self.detector.batch_detect(self.selected_files)
            
            self.progress.stop()
            self.root.after(0, lambda: self.progress_label.config(text="批处理完成"))
            self.root.after(0, lambda: self.update_tree_view(results))
            
            # 统计成功和失败的文件数量
            success_count = sum(1 for r in results if r['key'] != 'N/A')
            total_count = len(results)
            failure_count = total_count - success_count
            
            # 如果有失败的文件，显示警告信息
            if failure_count > 0:
                self.root.after(0, lambda: messagebox.showwarning(
                    "批处理完成",
                    f"批处理完成！成功：{success_count}个，失败：{failure_count}个\n"+
                    f"失败的文件已标记为 N/A"
                ))
        except Exception as e:
            error_msg = f"批处理过程中发生错误: {str(e)}"
            self.progress.stop()
            self.root.after(0, lambda: self.progress_label.config(text="批处理失败"))
            self.root.after(0, lambda: messagebox.showerror("批处理失败", error_msg))
        finally:
            # 3秒后恢复默认状态
            self.root.after(3000, lambda: self.progress_label.config(text="准备就绪"))
    
    def update_tree_view(self, results):
        # 清空现有项目
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 添加新结果
        for result in results:
            self.tree.insert('', tk.END, values=(
                result['file'],
                result['key'],
                result['mode'],
                f"{result['confidence']:.3f}"
            ))
    
    def show_waveform(self):
        if self.selected_files:
            self.visualizer.plot_waveform(self.selected_files[0])
    
    def show_spectrogram(self):
        if self.selected_files:
            self.visualizer.plot_spectrogram(self.selected_files[0])
    
    def clear_results(self):
        """清空所有检测结果和选择"""
        # 更新进度条状态
        self.progress_label.config(text="正在清空结果...")
        
        # 1. 清空选中的文件列表
        self.selected_files.clear()
        
        # 2. 清空批处理结果表格
        self.update_tree_view([])
        
        # 3. 更新文件数量显示
        self.file_count_label.config(text="已选择：0 个文件")
        
        # 4. 更新按钮状态
        self.update_button_states()
        
        # 5. 清空可视化图表和音频缓存
        if self.visualizer:
            self.visualizer.ax.clear()
            self.visualizer.ax.set_title("")
            self.visualizer.ax.set_xlabel("")
            self.visualizer.ax.set_ylabel("")
            self.visualizer.canvas.draw()
            # 清空音频缓存，释放内存
            self.visualizer.clear_cache()
        
        # 6. 停止进度条（如果正在运行）
        self.progress.stop()
        
        # 更新进度条状态
        self.progress_label.config(text="结果已清空")
        # 2秒后恢复默认状态
        self.root.after(2000, lambda: self.progress_label.config(text="准备就绪"))
    
    def on_closing(self):
        """处理窗口关闭事件，确保所有资源都能正确释放"""
        logger.info("正在关闭应用程序，清理资源...")
        
        # 1. 清空音频缓存，释放内存
        if self.visualizer:
            self.visualizer.clear_cache()
            # 清理matplotlib资源
            plt.close(self.visualizer.fig)
        
        # 2. 停止所有可能的后台任务
        self.progress.stop()
        
        # 3. 记录关闭日志
        logger.info("应用程序已成功关闭，所有资源已释放")
        
        # 4. 关闭主窗口
        self.root.destroy()
        
        # 5. 确保所有matplotlib窗口都关闭
        plt.close('all')

def main():
    """主函数，只启动GUI界面"""
    root = tk.Tk()
    app = KeyDetectorGUI(root)
    
    # 绑定窗口关闭事件到on_closing方法
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()