#!/usr/bin/env python3
"""
modularize_mediators_iteratively.py

内存高效的中介变量模块化分析脚本
采用字典驱动的迭代式和并行化方法处理3000+个高维、高缺失率的中介变量

作者: AI助手
日期: 2024
版本: 2.0
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
import pickle
from datetime import datetime
import psutil

# 科学计算和机器学习
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import font_manager

# 并行处理
from joblib import Parallel, delayed
import multiprocessing

# 抑制警告
warnings.filterwarnings('ignore')

# 修复中文字体显示问题
def setup_chinese_fonts():
    """设置中文字体"""
    try:
        # 尝试不同的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
        
        for font_name in chinese_fonts:
            try:
                # 检查字体是否可用
                font_path = font_manager.findfont(font_name, fallback_to_default=False)
                if font_path and 'default' not in font_path.lower():
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    print(f"✅ 成功设置中文字体: {font_name}")
                    break
            except:
                continue
        else:
            # 如果都找不到，使用系统默认字体并关闭中文显示
            print("⚠️ 未找到中文字体，使用英文显示")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        # 设置负号正常显示
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置字体大小
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['figure.titlesize'] = 14
        
    except Exception as e:
        print(f"⚠️ 字体设置失败: {e}")
        # 使用默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 初始化字体设置
setup_chinese_fonts()

class MemoryEfficientMediatorAnalyzer:
    """内存高效的中介变量分析器"""
    
    def __init__(self, 
                 variable_dict_path: str = "all_nhanes_variables.csv",
                 data_dir: str = "NHANES_PROCESSED_CSV_merged_by_prefix",
                 output_dir: str = "mediator_analysis_results_v2",
                 id_column: str = "RespondentSequenceNumber",
                 max_memory_gb: float = 100.0,  # 设置为100GB
                 n_jobs: int = 128):  # 设置为128，匹配CPU核心数
        """
        初始化分析器
        
        Args:
            variable_dict_path: 变量字典文件路径
            data_dir: 数据文件目录
            output_dir: 输出目录
            id_column: 参与者ID列名
            max_memory_gb: 最大内存使用限制(GB)
            n_jobs: 并行作业数
        """
        self.variable_dict_path = variable_dict_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.id_column = id_column
        self.max_memory_gb = max_memory_gb
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        # 设置日志（需要先创建基本目录）
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.setup_logging()
        
        # 创建完整输出目录结构
        self.create_output_directories()
        
        # 初始化状态变量
        self.variable_dict = None
        self.label_to_filepath_map = {}
        self.core_mediator_labels = []
        self.final_mediator_labels = []
        self.mediator_metadata = {}
        
        # 获取系统内存信息
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.logger.info(f"系统总内存: {total_memory:.1f}GB")
        self.logger.info(f"设置最大内存限制: {max_memory_gb}GB ({(max_memory_gb/total_memory*100):.1f}% of total)")
        self.logger.info(f"预留系统内存: {(total_memory - max_memory_gb):.1f}GB")
        self.logger.info(f"设置并行作业数: {self.n_jobs} (CPU核心总数: {multiprocessing.cpu_count()})")
        
        # 检查内存设置是否合理
        if max_memory_gb > total_memory * 0.95:
            self.logger.warning(f"⚠️ 内存限制 ({max_memory_gb}GB) 接近系统总内存 ({total_memory:.1f}GB)，可能影响系统稳定性")
        elif max_memory_gb < total_memory * 0.5:
            self.logger.warning(f"⚠️ 内存限制 ({max_memory_gb}GB) 可能过低，建议提高到系统总内存的70-85%之间")
        
        # 检查并行作业数设置是否合理
        if self.n_jobs > multiprocessing.cpu_count():
            self.logger.warning(f"⚠️ 并行作业数 ({self.n_jobs}) 超过CPU核心数 ({multiprocessing.cpu_count()})，可能影响性能")
        
        # 检查系统内存压力
        memory_pressure = psutil.virtual_memory().percent
        if memory_pressure > 80:
            self.logger.warning(f"⚠️ 系统内存压力较大 (使用率: {memory_pressure}%)，建议先清理一些内存再运行")
    
    def create_output_directories(self):
        """创建输出目录结构"""
        # 主输出目录
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 子目录
        self.correlation_dir = self.output_dir / "correlation_analysis"
        self.clustering_dir = self.output_dir / "clustering_results"
        self.visualization_dir = self.output_dir / "visualizations"
        self.module_dir = self.output_dir / "module_representatives"
        self.metadata_dir = self.output_dir / "metadata"
        
        # 创建所有子目录
        for dir_path in [self.correlation_dir, self.clustering_dir, self.visualization_dir, 
                        self.module_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        if hasattr(self, 'logger'):
            self.logger.info(f"📁 创建输出目录结构: {self.output_dir}")
        else:
            print(f"📁 创建输出目录结构: {self.output_dir}")
    
    def setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / f"mediator_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_memory_usage(self, operation: str = "") -> float:
        """检查内存使用情况"""
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.used / (1024**3)
        memory_percent = memory_info.percent
        
        # 更详细的内存使用报告
        if memory_gb > self.max_memory_gb:
            self.logger.warning(
                f"内存使用超限! {operation}\n"
                f"当前使用: {memory_gb:.2f}GB ({memory_percent:.1f}%)\n"
                f"最大限制: {self.max_memory_gb}GB\n"
                f"可用内存: {(memory_info.total/1024**3 - memory_gb):.2f}GB\n"
                f"内存压力: {'高' if memory_percent > 90 else '中等' if memory_percent > 70 else '正常'}"
            )
            gc.collect()  # 强制垃圾回收
        else:
            self.logger.debug(
                f"内存使用情况 {operation}:\n"
                f"当前使用: {memory_gb:.2f}GB ({memory_percent:.1f}%)\n"
                f"最大限制: {self.max_memory_gb}GB\n"
                f"内存余量: {(self.max_memory_gb - memory_gb):.2f}GB\n"
                f"内存压力: {'高' if memory_percent > 90 else '中等' if memory_percent > 70 else '正常'}"
            )
        
        return memory_gb
    
    def load_variable_dictionary(self) -> bool:
        """加载变量字典"""
        try:
            self.logger.info(f"📚 加载变量字典: {self.variable_dict_path}")
            self.variable_dict = pd.read_csv(self.variable_dict_path)
            
            required_cols = ['global_variable_id', 'file_name', 'variable_name', 'category']
            missing_cols = [col for col in required_cols if col not in self.variable_dict.columns]
            
            if missing_cols:
                self.logger.error(f"变量字典缺少必需列: {missing_cols}")
                return False
            
            self.logger.info(f"✅ 成功加载变量字典，共 {len(self.variable_dict)} 个变量")
            
            # 统计变量类型分布
            category_counts = self.variable_dict['category'].value_counts()
            self.logger.info("📊 变量类型分布:")
            for cat, count in category_counts.head(10).items():
                self.logger.info(f"   {cat}: {count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载变量字典失败: {e}")
            return False
    
    def build_label_to_filepath_map(self) -> bool:
        """构建标签到文件路径的映射"""
        try:
            self.logger.info("🗺️  构建标签到文件路径映射...")
            
            self.label_to_filepath_map = {}
            
            # 获取数据目录中所有CSV文件
            available_files = list(self.data_dir.glob("*.csv"))
            available_file_names = [f.name for f in available_files]
            
            self.logger.info(f"发现 {len(available_files)} 个数据文件")
            
            for _, row in self.variable_dict.iterrows():
                label = row['variable_name']
                dict_file_name = row['file_name']
                
                # 首先尝试精确匹配
                exact_path = self.data_dir / dict_file_name
                if exact_path.exists():
                    self.label_to_filepath_map[label] = exact_path
                    continue
                
                # 如果精确匹配失败，尝试智能匹配
                # 提取文件名的核心部分（去除第一个部分作为前缀）
                if '_' in dict_file_name:
                    core_name = '_'.join(dict_file_name.split('_')[1:])  # 去除第一个部分
                    
                    # 在可用文件中查找包含核心名称的文件，并验证变量存在
                    best_match = None
                    for available_file in available_file_names:
                        if core_name in available_file:
                            candidate_path = self.data_dir / available_file
                            
                            # 验证该文件是否真的包含这个变量
                            try:
                                # 只读取表头验证
                                header = pd.read_csv(candidate_path, nrows=0)
                                if label in header.columns:
                                    best_match = candidate_path
                                    break
                            except Exception:
                                continue
                    
                    if best_match:
                        self.label_to_filepath_map[label] = best_match
                    else:
                        # 如果验证失败，尝试不验证的匹配（兜底策略）
                        for available_file in available_file_names:
                            if core_name in available_file:
                                matched_path = self.data_dir / available_file
                                self.label_to_filepath_map[label] = matched_path
                                break
                        else:
                            self.logger.debug(f"文件未找到: {dict_file_name} -> {core_name}")
                else:
                    self.logger.debug(f"文件不存在: {exact_path}")
            
            self.logger.info(f"✅ 成功构建映射，覆盖 {len(self.label_to_filepath_map)} 个变量")
            
            # 统计匹配情况
            total_variables = len(self.variable_dict)
            matched_variables = len(self.label_to_filepath_map)
            match_rate = matched_variables / total_variables if total_variables > 0 else 0
            self.logger.info(f"📊 文件匹配率: {match_rate:.2%} ({matched_variables}/{total_variables})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 构建映射失败: {e}")
            return False
    
    def identify_core_mediators(self) -> bool:
        """识别核心中介变量"""
        try:
            self.logger.info("🔍 识别核心中介变量...")
            
            # 筛选B和B&C类型变量
            mediator_vars = self.variable_dict[
                self.variable_dict['category'].isin(['B', 'B&C'])
            ]
            
            # 提取变量名（标签）
            self.core_mediator_labels = mediator_vars['variable_name'].unique().tolist()
            
            # 过滤掉不存在文件路径的变量
            self.core_mediator_labels = [
                label for label in self.core_mediator_labels 
                if label in self.label_to_filepath_map
            ]
            
            self.logger.info(f"✅ 识别到 {len(self.core_mediator_labels)} 个核心中介变量")
            
            if len(self.core_mediator_labels) == 0:
                self.logger.error("❌ 没有找到有效的中介变量!")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 识别中介变量失败: {e}")
            return False
    
    def calculate_variable_metadata(self, label: str) -> Dict:
        """计算单个变量的元数据"""
        try:
            file_path = self.label_to_filepath_map[label]
            
            # 首先检查文件是否包含所需的列
            try:
                header = pd.read_csv(file_path, nrows=0)
                available_columns = header.columns.tolist()
                
                # 检查ID列是否存在
                if self.id_column not in available_columns:
                    # 如果没有标准ID列，尝试找第一列作为ID
                    if len(available_columns) > 0:
                        actual_id_col = available_columns[0]
                    else:
                        raise ValueError("文件没有任何列")
                else:
                    actual_id_col = self.id_column
                
                # 检查目标变量是否存在
                if label not in available_columns:
                    raise ValueError(f"变量 '{label}' 不在文件中")
                
                # 只加载ID列和目标变量列
                df = pd.read_csv(file_path, usecols=[actual_id_col, label], low_memory=False)
                
            except Exception as e:
                raise ValueError(f"文件读取失败: {e}")
            
            # 计算统计信息
            metadata = {
                'label': label,
                'file_path': str(file_path),
                'total_count': len(df),
                'missing_count': df[label].isna().sum(),
                'missing_rate': df[label].isna().mean(),
                'valid_count': df[label].notna().sum()
            }
            
            # 计算方差（只对数值型数据）
            valid_data = df[label].dropna()
            if len(valid_data) > 0:
                try:
                    # 尝试转换为数值型
                    numeric_data = pd.to_numeric(valid_data, errors='coerce').dropna()
                    if len(numeric_data) > 1:
                        metadata['variance'] = float(numeric_data.var())
                        metadata['mean'] = float(numeric_data.mean())
                        metadata['std'] = float(numeric_data.std())
                        metadata['is_numeric'] = True
                    else:
                        metadata['variance'] = 0.0
                        metadata['mean'] = np.nan
                        metadata['std'] = np.nan
                        metadata['is_numeric'] = False
                except:
                    metadata['variance'] = 0.0
                    metadata['mean'] = np.nan
                    metadata['std'] = np.nan
                    metadata['is_numeric'] = False
            else:
                metadata['variance'] = 0.0
                metadata['mean'] = np.nan
                metadata['std'] = np.nan
                metadata['is_numeric'] = False
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"计算变量 {label} 元数据失败: {e}")
            return {
                'label': label,
                'file_path': '',
                'total_count': 0,
                'missing_count': 0,
                'missing_rate': 1.0,
                'valid_count': 0,
                'variance': 0.0,
                'mean': np.nan,
                'std': np.nan,
                'is_numeric': False
            }
    
    def compute_metadata_iteratively(self, force_recalculate: bool = False) -> bool:
        """迭代计算所有中介变量的元数据"""
        try:
            metadata_path = self.metadata_dir / "mediator_metadata.csv"
            
            # 检查是否已存在元数据文件
            if metadata_path.exists() and not force_recalculate:
                self.logger.info("📋 发现已存在的元数据文件，直接加载...")
                try:
                    metadata_df = pd.read_csv(metadata_path)
                    self.logger.info(f"✅ 成功加载已存在的元数据，共 {len(metadata_df)} 个变量")
                    
                    # 转换为字典格式
                    self.mediator_metadata = {}
                    for _, row in metadata_df.iterrows():
                        self.mediator_metadata[row['label']] = row.to_dict()
                    
                    # 验证元数据完整性
                    existing_labels = set(self.mediator_metadata.keys())
                    expected_labels = set(self.core_mediator_labels)
                    
                    if existing_labels >= expected_labels:
                        self.logger.info("✅ 元数据完整，跳过重新计算")
                        
                        # 显示统计信息
                        missing_rate_stats = metadata_df['missing_rate'].describe()
                        self.logger.info("📈 缺失率统计:")
                        for stat, value in missing_rate_stats.items():
                            self.logger.info(f"   {stat}: {value:.4f}")
                        
                        return True
                    else:
                        missing_count = len(expected_labels - existing_labels)
                        self.logger.warning(f"⚠️  元数据不完整，缺少 {missing_count} 个变量，将重新计算")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️  加载已存在元数据失败: {e}，将重新计算")
            else:
                if force_recalculate:
                    self.logger.info("🔄 强制重新计算元数据...")
                else:
                    self.logger.info("📊 首次计算变量元数据...")
            
            # 执行元数据计算
            self.logger.info(f"使用 {self.n_jobs} 个进程并行计算...")
            
            # 修复joblib兼容性问题，使用更保守的参数
            try:
                results = Parallel(n_jobs=self.n_jobs, verbose=1, backend='threading')(
                    delayed(self.calculate_variable_metadata)(label) 
                    for label in self.core_mediator_labels
                )
            except Exception as e:
                self.logger.warning(f"并行计算失败，使用单线程: {e}")
                # 回退到单线程处理
                results = []
                for i, label in enumerate(self.core_mediator_labels):
                    if i % 100 == 0:
                        self.logger.info(f"处理进度: {i}/{len(self.core_mediator_labels)}")
                    result = self.calculate_variable_metadata(label)
                    results.append(result)
            
            # 转换为字典
            self.mediator_metadata = {result['label']: result for result in results}
            
            # 保存元数据
            metadata_df = pd.DataFrame(results)
            metadata_df.to_csv(metadata_path, index=False)
            self.logger.info(f"💾 元数据已保存到: {metadata_path}")
            
            # 统计信息
            missing_rate_stats = metadata_df['missing_rate'].describe()
            self.logger.info("📈 缺失率统计:")
            for stat, value in missing_rate_stats.items():
                self.logger.info(f"   {stat}: {value:.4f}")
            
            # 检查内存
            self.check_memory_usage("计算元数据后")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 计算元数据失败: {e}")
            return False
    
    def filter_variables(self, 
                        max_missing_rate: float = 0.9, 
                        min_variance: float = 1e-8) -> bool:
        """根据质量标准筛选变量"""
        try:
            self.logger.info(f"🔬 筛选变量 (缺失率≤{max_missing_rate}, 方差≥{min_variance})...")
            
            original_count = len(self.core_mediator_labels)
            
            self.final_mediator_labels = []
            filtered_reasons = {'high_missing': 0, 'low_variance': 0, 'non_numeric': 0}
            
            for label in self.core_mediator_labels:
                metadata = self.mediator_metadata[label]
                
                # 检查缺失率
                if metadata['missing_rate'] > max_missing_rate:
                    filtered_reasons['high_missing'] += 1
                    continue
                
                # 检查是否为数值型
                if not metadata['is_numeric']:
                    filtered_reasons['non_numeric'] += 1
                    continue
                
                # 检查方差
                if metadata['variance'] < min_variance:
                    filtered_reasons['low_variance'] += 1
                    continue
                
                self.final_mediator_labels.append(label)
            
            self.logger.info(f"✅ 筛选完成: {original_count} → {len(self.final_mediator_labels)}")
            self.logger.info(f"   过滤原因: 高缺失={filtered_reasons['high_missing']}, "
                           f"低方差={filtered_reasons['low_variance']}, "
                           f"非数值={filtered_reasons['non_numeric']}")
            
            if len(self.final_mediator_labels) < 2:
                self.logger.error("❌ 有效变量数量不足，无法进行相关性分析!")
                return False
            
            # 保存最终变量列表
            final_vars_path = self.metadata_dir / "final_mediator_variables.txt"
            with open(final_vars_path, 'w', encoding='utf-8') as f:
                for label in self.final_mediator_labels:
                    f.write(f"{label}\n")
            
            self.logger.info(f"💾 最终变量列表已保存到: {final_vars_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 变量筛选失败: {e}")
            return False
    
    def get_id_column_for_file(self, file_path: Path) -> str:
        """获取文件的实际ID列名"""
        try:
            header = pd.read_csv(file_path, nrows=0)
            columns = header.columns.tolist()
            
            # 首先尝试标准ID列
            if self.id_column in columns:
                return self.id_column
            
            # 如果没有标准ID列，使用第一列
            if len(columns) > 0:
                return columns[0]
            
            raise ValueError("文件没有任何列")
            
        except Exception as e:
            raise ValueError(f"无法确定ID列: {e}")
    
    def calculate_pairwise_correlation(self, label1: str, label2: str) -> Tuple[int, int, float]:
        """计算两个变量的相关系数"""
        try:
            # 获取文件路径
            file_path1 = self.label_to_filepath_map[label1]
            file_path2 = self.label_to_filepath_map[label2]
            
            # 获取每个文件的实际ID列名
            id_col1 = self.get_id_column_for_file(file_path1)
            id_col2 = self.get_id_column_for_file(file_path2)
            
            # 加载数据
            if file_path1 == file_path2:
                # 同一文件，一次性加载
                # 检查所有需要的列是否存在
                header = pd.read_csv(file_path1, nrows=0)
                available_cols = header.columns.tolist()
                
                if id_col1 not in available_cols or label1 not in available_cols or label2 not in available_cols:
                    raise ValueError(f"文件中缺少必需的列")
                
                df = pd.read_csv(file_path1, usecols=[id_col1, label1, label2], low_memory=False)
                df1 = df[[id_col1, label1]].copy()
                df2 = df[[id_col1, label2]].copy()
                
                # 重命名ID列以便合并
                df1.rename(columns={id_col1: 'ID'}, inplace=True)
                df2.rename(columns={id_col1: 'ID'}, inplace=True)
                
            else:
                # 不同文件，分别加载
                # 检查第一个文件
                header1 = pd.read_csv(file_path1, nrows=0)
                if id_col1 not in header1.columns or label1 not in header1.columns:
                    raise ValueError(f"文件1中缺少必需的列: {[id_col1, label1]}")
                
                # 检查第二个文件  
                header2 = pd.read_csv(file_path2, nrows=0)
                if id_col2 not in header2.columns or label2 not in header2.columns:
                    raise ValueError(f"文件2中缺少必需的列: {[id_col2, label2]}")
                
                df1 = pd.read_csv(file_path1, usecols=[id_col1, label1], low_memory=False)
                df2 = pd.read_csv(file_path2, usecols=[id_col2, label2], low_memory=False)
                
                # 重命名ID列以便合并
                df1.rename(columns={id_col1: 'ID'}, inplace=True)
                df2.rename(columns={id_col2: 'ID'}, inplace=True)
            
            # 合并数据
            merged_df = pd.merge(df1, df2, on='ID', how='inner')
            
            # 移除缺失值
            merged_df = merged_df.dropna()
            
            if len(merged_df) < 50:  # 要求至少50个有效观测
                return self.final_mediator_labels.index(label1), self.final_mediator_labels.index(label2), 0.0
            
            # 转换为数值型
            x = pd.to_numeric(merged_df[label1], errors='coerce')
            y = pd.to_numeric(merged_df[label2], errors='coerce')
            
            # 再次移除转换失败的值
            valid_mask = ~(x.isna() | y.isna())
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) < 50:
                return self.final_mediator_labels.index(label1), self.final_mediator_labels.index(label2), 0.0
            
            # 计算皮尔逊相关系数
            correlation, _ = pearsonr(x, y)
            
            # 处理NaN值
            if np.isnan(correlation):
                correlation = 0.0
            
            return self.final_mediator_labels.index(label1), self.final_mediator_labels.index(label2), correlation
            
        except Exception as e:
            self.logger.warning(f"计算相关性失败 {label1} vs {label2}: {e}")
            return self.final_mediator_labels.index(label1), self.final_mediator_labels.index(label2), 0.0
    
    def build_correlation_matrix_parallel(self) -> bool:
        """并行构建相关性矩阵"""
        try:
            self.logger.info("🔗 开始并行构建相关性矩阵...")
            
            n_vars = len(self.final_mediator_labels)
            self.logger.info(f"变量数量: {n_vars}, 需要计算 {n_vars*(n_vars-1)//2} 个变量对")
            
            # 初始化相关性矩阵
            corr_matrix = np.eye(n_vars)  # 对角线为1
            
            # 创建变量对
            variable_pairs = []
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    variable_pairs.append((self.final_mediator_labels[i], self.final_mediator_labels[j]))
            
            # 分批处理
            batch_size = 10000
            n_batches = (len(variable_pairs) + batch_size - 1) // batch_size
            self.logger.info(f"将计算分为 {n_batches} 个批次，每批最多 {batch_size} 个对")
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(variable_pairs))
                batch_pairs = variable_pairs[start:end]
                
                self.logger.info(f"处理批次 {batch_idx+1}/{n_batches} ({len(batch_pairs)} 个对)")
                
                # 并行计算批次
                results = Parallel(n_jobs=self.n_jobs, verbose=10, batch_size=100)(
                    delayed(self.calculate_pairwise_correlation)(label1, label2) 
                    for label1, label2 in batch_pairs
                )
                
                # 更新矩阵
                for i, j, corr in results:
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                
                # 保存中间结果
                intermediate_path = self.correlation_dir / f"correlation_matrix_batch_{batch_idx+1}.npy"
                np.save(intermediate_path, corr_matrix)
                self.logger.info(f"💾 中间相关性矩阵保存到: {intermediate_path}")
                
                # 检查内存
                self.check_memory_usage(f"批次 {batch_idx+1} 完成后")
            
            # 保存最终结果
            corr_matrix_path = self.correlation_dir / "correlation_matrix.npy"
            np.save(corr_matrix_path, corr_matrix)
            self.logger.info(f"💾 最终相关性矩阵已保存到: {corr_matrix_path}")
            
            # 保存为CSV
            corr_df = pd.DataFrame(corr_matrix, 
                                 index=self.final_mediator_labels, 
                                 columns=self.final_mediator_labels)
            corr_csv_path = self.correlation_dir / "correlation_matrix.csv"
            corr_df.to_csv(corr_csv_path)
            
            # 统计
            upper_triangle = corr_matrix[np.triu_indices(n_vars, k=1)]
            corr_stats = pd.Series(upper_triangle).describe()
            self.logger.info("📈 相关性统计:")
            for stat, value in corr_stats.items():
                self.logger.info(f"   {stat}: {value:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 构建相关性矩阵失败: {e}")
            return False
    
    def perform_hierarchical_clustering(self, 
                                      linkage_method: str = 'ward',
                                      n_clusters: int = 20) -> bool:
        """执行层次聚类"""
        try:
            self.logger.info(f"🌳 开始层次聚类 (方法: {linkage_method}, 簇数: {n_clusters})...")
            
            # 加载相关性矩阵
            corr_matrix_path = self.correlation_dir / "correlation_matrix.npy"
            corr_matrix = np.load(corr_matrix_path)
            
            # 转换为距离矩阵
            distance_matrix = 1 - np.abs(corr_matrix)
            
            # 确保距离矩阵的对角线为0
            np.fill_diagonal(distance_matrix, 0)
            
            # 将距离矩阵转换为压缩形式
            from scipy.spatial.distance import squareform
            distance_vector = squareform(distance_matrix, checks=False)
            
            # 执行层次聚类
            self.logger.info("🔗 执行层次聚类...")
            linkage_matrix = linkage(distance_vector, method=linkage_method)
            
            # 生成树状图
            self.logger.info("🎨 生成树状图...")
            plt.figure(figsize=(20, 10))
            dendrogram(linkage_matrix, 
                      labels=self.final_mediator_labels,
                      leaf_rotation=90,
                      leaf_font_size=8)
            plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method})')
            plt.xlabel('Variables')
            plt.ylabel('Distance')
            plt.tight_layout()
            
            dendrogram_path = self.visualization_dir / "dendrogram_correlation.png"
            plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"🖼️  树状图已保存到: {dendrogram_path}")
            
            # 生成簇标签
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # 创建模块映射
            module_mapping = pd.DataFrame({
                'Label': self.final_mediator_labels,
                'ModuleID': cluster_labels
            })
            
            # 保存模块映射
            mapping_path = self.clustering_dir / "mediator_module_mapping_v2.csv"
            module_mapping.to_csv(mapping_path, index=False)
            self.logger.info(f"💾 模块映射已保存到: {mapping_path}")
            
            # 统计模块信息
            module_stats = module_mapping['ModuleID'].value_counts().sort_index()
            self.logger.info("📊 模块统计:")
            for module_id, count in module_stats.items():
                self.logger.info(f"   模块 {module_id}: {count} 个变量")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 层次聚类失败: {e}")
            return False
    
    def load_module_data(self, module_labels: List[str]) -> pd.DataFrame:
        """加载模块中所有变量的数据"""
        try:
            # 按文件分组变量
            file_groups = {}
            for label in module_labels:
                file_path = self.label_to_filepath_map[label]
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append(label)
            
            # 加载并合并数据
            module_dfs = []
            for file_path, labels_in_file in file_groups.items():
                # 获取实际的ID列名
                id_col = self.get_id_column_for_file(file_path)
                
                # 检查所有列是否存在
                header = pd.read_csv(file_path, nrows=0)
                available_cols = header.columns.tolist()
                
                # 过滤存在的列
                valid_labels = [label for label in labels_in_file if label in available_cols]
                if not valid_labels:
                    continue
                
                columns_to_load = [id_col] + valid_labels
                df = pd.read_csv(file_path, usecols=columns_to_load, low_memory=False)
                
                # 重命名ID列为标准名称
                df.rename(columns={id_col: 'ID'}, inplace=True)
                module_dfs.append(df)
            
            # 合并所有数据
            if len(module_dfs) == 0:
                return pd.DataFrame()
            elif len(module_dfs) == 1:
                merged_df = module_dfs[0]
            else:
                merged_df = module_dfs[0]
                for df in module_dfs[1:]:
                    merged_df = pd.merge(merged_df, df, on='ID', how='outer')
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"加载模块数据失败: {e}")
            return pd.DataFrame()
    
    def generate_module_representatives(self, 
                                      n_components: int = 1,
                                      imputation_neighbors: int = 5) -> bool:
        """生成模块代表"""
        try:
            self.logger.info("🧬 开始生成模块代表...")
            
            # 加载模块映射
            mapping_path = self.clustering_dir / "mediator_module_mapping_v2.csv"
            module_mapping = pd.read_csv(mapping_path)
            
            # 初始化最终结果DataFrame（只包含ID）
            # 我们需要获取所有参与者的ID
            self.logger.info("📋 获取所有参与者ID...")
            
            # 从第一个数据文件获取完整的ID列表
            first_file = next(iter(self.label_to_filepath_map.values()))
            first_file_id_col = self.get_id_column_for_file(first_file)
            all_ids_df = pd.read_csv(first_file, usecols=[first_file_id_col], low_memory=False)
            # 重命名为标准ID列名
            all_ids_df.rename(columns={first_file_id_col: 'ID'}, inplace=True)
            final_representatives = all_ids_df.copy()
            
            # 按模块处理
            unique_modules = sorted(module_mapping['ModuleID'].unique())
            self.logger.info(f"处理 {len(unique_modules)} 个模块...")
            
            for module_idx, module_id in enumerate(unique_modules):
                try:
                    self.logger.info(f"🔬 处理模块 {module_id} ({module_idx+1}/{len(unique_modules)})...")
                    
                    # 获取模块中的变量
                    module_labels = module_mapping[
                        module_mapping['ModuleID'] == module_id
                    ]['Label'].tolist()
                    
                    self.logger.info(f"   模块 {module_id} 包含 {len(module_labels)} 个变量")
                    
                    # 加载模块数据
                    module_df = self.load_module_data(module_labels)
                    
                    if module_df.empty:
                        self.logger.warning(f"   模块 {module_id} 数据加载失败，跳过")
                        continue
                    
                    # 检查内存
                    self.check_memory_usage(f"加载模块 {module_id} 后")
                    
                    # 准备数据用于PCA
                    feature_columns = [col for col in module_df.columns if col != 'ID']
                    X = module_df[feature_columns].copy()
                    
                    # 转换为数值型
                    self.logger.info(f"   模块 {module_id}: 转换数据类型...")
                    for col in feature_columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    
                    # 检查有效数据
                    valid_samples = X.notna().any(axis=1)
                    if valid_samples.sum() < 50:
                        self.logger.warning(f"   模块 {module_id} 有效样本不足({valid_samples.sum()})，跳过")
                        continue
                    
                    # 缺失值填补
                    self.logger.info(f"   模块 {module_id}: 开始KNN缺失值填补...")
                    self.logger.info(f"   数据维度: {X.shape}, 缺失率: {X.isna().sum().sum() / (X.shape[0] * X.shape[1]):.2%}")
                    
                    # 计算合适的邻居数
                    effective_neighbors = min(imputation_neighbors, valid_samples.sum()-1, 10)
                    self.logger.info(f"   使用KNN邻居数: {effective_neighbors}")
                    
                    try:
                        imputer = KNNImputer(n_neighbors=effective_neighbors)
                        
                        # 分块处理大数据以显示进度
                        if X.shape[0] > 10000:
                            self.logger.info(f"   大数据集检测，分块处理...")
                            chunk_size = 5000
                            n_chunks = (X.shape[0] + chunk_size - 1) // chunk_size
                            
                            X_imputed_chunks = []
                            for chunk_idx in range(n_chunks):
                                start_idx = chunk_idx * chunk_size
                                end_idx = min(start_idx + chunk_size, X.shape[0])
                                
                                self.logger.info(f"   KNN填补进度: 块 {chunk_idx+1}/{n_chunks} ({start_idx}:{end_idx})")
                                
                                X_chunk = X.iloc[start_idx:end_idx]
                                X_chunk_imputed = imputer.fit_transform(X_chunk)
                                X_imputed_chunks.append(X_chunk_imputed)
                            
                            X_imputed_array = np.vstack(X_imputed_chunks)
                        else:
                            self.logger.info(f"   执行KNN填补...")
                            X_imputed_array = imputer.fit_transform(X)
                        
                        X_imputed = pd.DataFrame(X_imputed_array, columns=feature_columns, index=X.index)
                        self.logger.info(f"   KNN填补完成")
                        
                    except Exception as impute_error:
                        self.logger.warning(f"   KNN填补失败: {impute_error}, 使用均值填补")
                        X_imputed = X.fillna(X.mean())
                    
                    # 标准化
                    self.logger.info(f"   模块 {module_id}: 标准化...")
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_imputed)
                    
                    # PCA
                    self.logger.info(f"   模块 {module_id}: PCA降维...")
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # 创建主成分DataFrame
                    pc_column = f"Module_{module_id}_PC1"
                    pc_df = pd.DataFrame({
                        'ID': module_df['ID'],
                        pc_column: X_pca[:, 0]
                    })
                    
                    # 合并到最终结果
                    final_representatives = pd.merge(
                        final_representatives, pc_df, 
                        on='ID', how='left'
                    )
                    
                    # 保存中间结果
                    intermediate_path = self.module_dir / "module_representatives_partial.parquet"
                    final_representatives.to_parquet(intermediate_path, index=False)
                    self.logger.info(f"💾 中间模块代表保存到: {intermediate_path}")
                    
                    # 记录PCA信息
                    explained_variance = pca.explained_variance_ratio_[0]
                    self.logger.info(f"   模块 {module_id}: PC1解释方差比 = {explained_variance:.4f}")
                    self.logger.info(f"   完成进度: {module_idx+1}/{len(unique_modules)} ({(module_idx+1)/len(unique_modules)*100:.1f}%)")
                    
                    # 清理内存
                    del module_df, X, X_imputed, X_scaled, X_pca, pc_df
                    gc.collect()
                    
                except Exception as e:
                    self.logger.error(f"   处理模块 {module_id} 失败: {e}")
                    continue
            
            # 保存最终结果
            output_path = self.module_dir / "module_representatives_pca_v2.parquet"
            final_representatives.to_parquet(output_path, index=False)
            self.logger.info(f"💾 最终模块代表已保存到: {output_path}")
            
            # 统计信息
            n_modules_generated = len([col for col in final_representatives.columns if col.startswith("Module_")])
            self.logger.info(f"✅ 成功生成 {n_modules_generated} 个模块代表")
            
            # 最终内存检查
            self.check_memory_usage("生成模块代表完成后")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 生成模块代表失败: {e}")
            return False
    
    def visualize_correlation_matrix(self) -> bool:
        """可视化相关性矩阵"""
        try:
            self.logger.info("🎨 生成相关性矩阵可视化...")
            
            # 加载相关性矩阵
            corr_matrix_path = self.correlation_dir / "correlation_matrix.npy"
            corr_matrix = np.load(corr_matrix_path)
            
            # 创建大图
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('中介变量相关性矩阵分析', fontsize=16, fontweight='bold')
            
            # 1. 相关性分布直方图
            upper_triangle = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
            axes[0, 0].hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('相关性分布直方图')
            axes[0, 0].set_xlabel('相关系数')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 相关性热力图（采样显示）
            sample_size = min(100, corr_matrix.shape[0])
            sample_indices = np.linspace(0, corr_matrix.shape[0]-1, sample_size, dtype=int)
            sample_matrix = corr_matrix[sample_indices][:, sample_indices]
            
            im = axes[0, 1].imshow(sample_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, 1].set_title(f'相关性热力图 (采样{sample_size}个变量)')
            axes[0, 1].set_xlabel('变量索引')
            axes[0, 1].set_ylabel('变量索引')
            plt.colorbar(im, ax=axes[0, 1])
            
            # 3. 相关性统计箱线图
            # 计算每个变量的平均相关性
            mean_correlations = np.mean(np.abs(corr_matrix), axis=1)
            axes[1, 0].boxplot(mean_correlations)
            axes[1, 0].set_title('变量平均相关性分布')
            axes[1, 0].set_ylabel('平均绝对相关系数')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 相关性强度分布
            # 按相关性强度分类
            strong_corr = np.sum(np.abs(upper_triangle) > 0.7)
            moderate_corr = np.sum((np.abs(upper_triangle) > 0.3) & (np.abs(upper_triangle) <= 0.7))
            weak_corr = np.sum((np.abs(upper_triangle) > 0.1) & (np.abs(upper_triangle) <= 0.3))
            very_weak_corr = np.sum(np.abs(upper_triangle) <= 0.1)
            
            categories = ['强相关\n(>0.7)', '中等相关\n(0.3-0.7)', '弱相关\n(0.1-0.3)', '极弱相关\n(≤0.1)']
            counts = [strong_corr, moderate_corr, weak_corr, very_weak_corr]
            colors = ['red', 'orange', 'yellow', 'lightblue']
            
            bars = axes[1, 1].bar(categories, counts, color=colors, alpha=0.7)
            axes[1, 1].set_title('相关性强度分布')
            axes[1, 1].set_ylabel('变量对数量')
            
            # 在柱状图上添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{count:,}\n({count/len(upper_triangle)*100:.1f}%)',
                               ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图片
            viz_path = self.visualization_dir / "correlation_matrix_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"💾 相关性矩阵可视化已保存到: {viz_path}")
            
            # 保存相关性统计到CSV
            corr_stats = {
                '统计指标': ['总变量对数量', '强相关(>0.7)', '中等相关(0.3-0.7)', '弱相关(0.1-0.3)', '极弱相关(≤0.1)'],
                '数量': [len(upper_triangle), strong_corr, moderate_corr, weak_corr, very_weak_corr],
                '百分比': [100, strong_corr/len(upper_triangle)*100, moderate_corr/len(upper_triangle)*100, 
                          weak_corr/len(upper_triangle)*100, very_weak_corr/len(upper_triangle)*100]
            }
            corr_stats_df = pd.DataFrame(corr_stats)
            corr_stats_path = self.correlation_dir / "correlation_statistics.csv"
            corr_stats_df.to_csv(corr_stats_path, index=False)
            
            self.logger.info(f"💾 相关性统计已保存到: {corr_stats_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 相关性矩阵可视化失败: {e}")
            return False
    
    def visualize_clustering_results(self) -> bool:
        """可视化聚类结果"""
        try:
            self.logger.info("🎨 生成聚类结果可视化...")
            
            # 加载模块映射
            mapping_path = self.clustering_dir / "mediator_module_mapping_v2.csv"
            module_mapping = pd.read_csv(mapping_path)
            
            # 创建大图
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('中介变量聚类分析结果', fontsize=16, fontweight='bold')
            
            # 1. 模块大小分布
            module_sizes = module_mapping['ModuleID'].value_counts().sort_index()
            axes[0, 0].bar(range(1, len(module_sizes)+1), module_sizes.values, alpha=0.7, color='lightcoral')
            axes[0, 0].set_title('各模块变量数量分布')
            axes[0, 0].set_xlabel('模块ID')
            axes[0, 0].set_ylabel('变量数量')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, size in enumerate(module_sizes.values):
                axes[0, 0].text(i+1, size, str(size), ha='center', va='bottom')
            
            # 2. 模块大小饼图
            sizes = module_sizes.values
            labels = [f'模块{i}' for i in module_sizes.index]
            colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
            
            axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('模块大小比例分布')
            
            # 3. 模块大小统计
            size_stats = module_sizes.describe()
            axes[1, 0].text(0.1, 0.9, f"模块统计信息:", fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)
            y_pos = 0.8
            for stat, value in size_stats.items():
                axes[1, 0].text(0.1, y_pos, f"{stat}: {value:.1f}", fontsize=10, transform=axes[1, 0].transAxes)
                y_pos -= 0.1
            
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('模块大小统计')
            axes[1, 0].axis('off')
            
            # 4. 模块大小分布直方图
            axes[1, 1].hist(module_sizes.values, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 1].set_title('模块大小分布直方图')
            axes[1, 1].set_xlabel('模块大小')
            axes[1, 1].set_ylabel('频数')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            viz_path = self.visualization_dir / "clustering_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"💾 聚类结果可视化已保存到: {viz_path}")
            
            # 保存模块统计到CSV
            module_stats = pd.DataFrame({
                '模块ID': module_sizes.index,
                '变量数量': module_sizes.values,
                '百分比': module_sizes.values / len(module_mapping) * 100
            })
            module_stats_path = self.clustering_dir / "module_statistics.csv"
            module_stats.to_csv(module_stats_path, index=False)
            
            self.logger.info(f"💾 模块统计已保存到: {module_stats_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 聚类结果可视化失败: {e}")
            return False
    
    def visualize_module_representatives(self) -> bool:
        """可视化模块代表"""
        try:
            self.logger.info("🎨 生成模块代表可视化...")
            
            # 加载模块代表数据
            partial_path = self.module_dir / "module_representatives_partial.parquet"
            if not partial_path.exists():
                self.logger.warning("⚠️ 模块代表文件不存在，跳过可视化")
                return False
            
            module_data = pd.read_parquet(partial_path)
            
            # 获取模块列
            module_columns = [col for col in module_data.columns if col.startswith('Module_')]
            if not module_columns:
                self.logger.warning("⚠️ 没有找到模块代表列，跳过可视化")
                return False
            
            # 创建大图
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('模块代表分析', fontsize=16, fontweight='bold')
            
            # 1. 模块代表分布
            for i, col in enumerate(module_columns[:5]):  # 只显示前5个模块
                valid_data = module_data[col].dropna()
                if len(valid_data) > 0:
                    axes[0, 0].hist(valid_data, bins=30, alpha=0.6, label=f'模块{col.split("_")[1]}')
            
            axes[0, 0].set_title('模块代表分布 (前5个模块)')
            axes[0, 0].set_xlabel('模块代表值')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 模块代表箱线图
            plot_data = []
            plot_labels = []
            for col in module_columns[:10]:  # 只显示前10个模块
                valid_data = module_data[col].dropna()
                if len(valid_data) > 0:
                    plot_data.append(valid_data)
                    plot_labels.append(f'模块{col.split("_")[1]}')
            
            if plot_data:
                axes[0, 1].boxplot(plot_data, labels=plot_labels)
                axes[0, 1].set_title('模块代表箱线图 (前10个模块)')
                axes[0, 1].set_ylabel('模块代表值')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 模块代表相关性热力图
            if len(module_columns) > 1:
                corr_matrix = module_data[module_columns].corr()
                im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[1, 0].set_title('模块代表相关性热力图')
                axes[1, 0].set_xticks(range(len(module_columns)))
                axes[1, 0].set_yticks(range(len(module_columns)))
                axes[1, 0].set_xticklabels([f'M{col.split("_")[1]}' for col in module_columns], rotation=45)
                axes[1, 0].set_yticklabels([f'M{col.split("_")[1]}' for col in module_columns])
                plt.colorbar(im, ax=axes[1, 0])
            
            # 4. 模块代表统计信息
            stats_text = f"总模块数: {len(module_columns)}\n"
            stats_text += f"总参与者数: {len(module_data)}\n\n"
            
            for col in module_columns[:5]:  # 只显示前5个模块的统计
                valid_data = module_data[col].dropna()
                if len(valid_data) > 0:
                    stats_text += f"{col}:\n"
                    stats_text += f"  有效样本: {len(valid_data)}\n"
                    stats_text += f"  均值: {valid_data.mean():.3f}\n"
                    stats_text += f"  标准差: {valid_data.std():.3f}\n\n"
            
            axes[1, 1].text(0.1, 0.9, stats_text, fontsize=10, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('模块代表统计信息')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # 保存图片
            viz_path = self.visualization_dir / "module_representatives_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"💾 模块代表可视化已保存到: {viz_path}")
            
            # 保存模块代表统计到CSV
            module_stats = []
            for col in module_columns:
                valid_data = module_data[col].dropna()
                if len(valid_data) > 0:
                    module_stats.append({
                        '模块ID': col.split('_')[1],
                        '有效样本数': len(valid_data),
                        '均值': valid_data.mean(),
                        '标准差': valid_data.std(),
                        '最小值': valid_data.min(),
                        '最大值': valid_data.max(),
                        '缺失率': (len(module_data) - len(valid_data)) / len(module_data) * 100
                    })
            
            if module_stats:
                module_stats_df = pd.DataFrame(module_stats)
                module_stats_path = self.module_dir / "module_representatives_statistics.csv"
                module_stats_df.to_csv(module_stats_path, index=False)
                self.logger.info(f"💾 模块代表统计已保存到: {module_stats_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模块代表可视化失败: {e}")
            return False

    def run_complete_analysis(self, 
                            max_missing_rate: float = 0.9,
                            min_variance: float = 1e-8,
                            n_clusters: int = 20,
                            linkage_method: str = 'ward',
                            force_recalculate_metadata: bool = False,
                            start_from_stage: int = 1) -> bool:
        """运行完整的分析流程"""
        try:
            self.logger.info("🚀 开始完整的中介变量模块化分析...")
            
            if start_from_stage <= 1:
                # 第一阶段：元数据预计算与智能查找表构建
                self.logger.info("\n" + "="*60)
                self.logger.info("第一阶段：元数据预计算与智能查找表构建")
                self.logger.info("="*60)
                
                if not self.load_variable_dictionary():
                    return False
                
                if not self.build_label_to_filepath_map():
                    return False
                
                if not self.identify_core_mediators():
                    return False
                
                if not self.compute_metadata_iteratively(force_recalculate_metadata):
                    return False
                
                if not self.filter_variables(max_missing_rate, min_variance):
                    return False
            else:
                self.logger.info("⏭️ 跳过第一阶段（元数据预计算），加载已有结果...")
                # 加载必要的数据结构
                if not self.load_existing_results_stage1():
                    return False
            
            if start_from_stage <= 2:
                # 第二阶段：迭代式并行构建相关性矩阵
                self.logger.info("\n" + "="*60)
                self.logger.info("第二阶段：迭代式并行构建相关性矩阵")
                self.logger.info("="*60)
                
                if not self.build_correlation_matrix_parallel():
                    return False
                
                # 可视化相关性矩阵
                if not self.visualize_correlation_matrix():
                    self.logger.warning("⚠️ 相关性矩阵可视化失败，继续执行")
            else:
                self.logger.info("⏭️ 跳过第二阶段（相关性矩阵构建），检查已有结果...")
                if not self.check_correlation_matrix_exists():
                    return False
            
            if start_from_stage <= 3:
                # 第三阶段：在相关性矩阵上进行聚类
                self.logger.info("\n" + "="*60)
                self.logger.info("第三阶段：在相关性矩阵上进行聚类")
                self.logger.info("="*60)
                
                if not self.perform_hierarchical_clustering(linkage_method, n_clusters):
                    return False
                
                # 可视化聚类结果
                if not self.visualize_clustering_results():
                    self.logger.warning("⚠️ 聚类结果可视化失败，继续执行")
            else:
                self.logger.info("⏭️ 跳过第三阶段（聚类分析），检查已有结果...")
                if not self.check_clustering_results_exist():
                    return False
            
            # 第四阶段：迭代式生成模块代表
            self.logger.info("\n" + "="*60)
            self.logger.info("第四阶段：迭代式生成模块代表")
            self.logger.info("="*60)
            
            if not self.generate_module_representatives():
                return False
            
            # 可视化模块代表
            if not self.visualize_module_representatives():
                self.logger.warning("⚠️ 模块代表可视化失败，继续执行")
            
            # 分析完成
            self.logger.info("\n" + "="*60)
            self.logger.info("🎉 分析完成!")
            self.logger.info("="*60)
            self.logger.info(f"📁 所有结果保存在: {self.output_dir}")
            self.logger.info("📋 主要输出文件:")
            self.logger.info(f"   - 变量元数据: {self.metadata_dir}/mediator_metadata.csv")
            self.logger.info(f"   - 最终变量列表: {self.metadata_dir}/final_mediator_variables.txt")
            self.logger.info(f"   - 相关性矩阵: {self.correlation_dir}/correlation_matrix.npy/.csv")
            self.logger.info(f"   - 相关性统计: {self.correlation_dir}/correlation_statistics.csv")
            self.logger.info(f"   - 聚类树状图: {self.visualization_dir}/dendrogram_correlation.png")
            self.logger.info(f"   - 聚类分析: {self.visualization_dir}/clustering_analysis.png")
            self.logger.info(f"   - 模块映射: {self.clustering_dir}/mediator_module_mapping_v2.csv")
            self.logger.info(f"   - 模块统计: {self.clustering_dir}/module_statistics.csv")
            self.logger.info(f"   - 模块代表: {self.module_dir}/module_representatives_pca_v2.parquet")
            self.logger.info(f"   - 模块代表统计: {self.module_dir}/module_representatives_statistics.csv")
            self.logger.info(f"   - 相关性分析: {self.visualization_dir}/correlation_matrix_analysis.png")
            self.logger.info(f"   - 模块代表分析: {self.visualization_dir}/module_representatives_analysis.png")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 完整分析失败: {e}")
            return False
    
    def load_existing_results_stage1(self) -> bool:
        """加载第一阶段的已有结果"""
        try:
            self.logger.info("📋 加载第一阶段已有结果...")
            
            # 加载变量字典
            if not self.load_variable_dictionary():
                return False
            
            # 构建映射
            if not self.build_label_to_filepath_map():
                return False
            
            # 加载元数据
            metadata_path = self.metadata_dir / "mediator_metadata.csv"
            if not metadata_path.exists():
                self.logger.error(f"❌ 元数据文件不存在: {metadata_path}")
                return False
            
            metadata_df = pd.read_csv(metadata_path)
            self.mediator_metadata = {}
            for _, row in metadata_df.iterrows():
                self.mediator_metadata[row['label']] = row.to_dict()
            
            # 加载最终变量列表
            final_vars_path = self.metadata_dir / "final_mediator_variables.txt"
            if not final_vars_path.exists():
                self.logger.error(f"❌ 最终变量列表不存在: {final_vars_path}")
                return False
            
            with open(final_vars_path, 'r', encoding='utf-8') as f:
                self.final_mediator_labels = [line.strip() for line in f if line.strip()]
            
            self.logger.info(f"✅ 成功加载第一阶段结果，共 {len(self.final_mediator_labels)} 个变量")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载第一阶段结果失败: {e}")
            return False
    
    def check_correlation_matrix_exists(self) -> bool:
        """检查相关性矩阵是否存在"""
        try:
            corr_matrix_path = self.correlation_dir / "correlation_matrix.npy"
            if not corr_matrix_path.exists():
                self.logger.error(f"❌ 相关性矩阵不存在: {corr_matrix_path}")
                return False
            
            self.logger.info(f"✅ 相关性矩阵已存在: {corr_matrix_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 检查相关性矩阵失败: {e}")
            return False
    
    def check_clustering_results_exist(self) -> bool:
        """检查聚类结果是否存在"""
        try:
            mapping_path = self.clustering_dir / "mediator_module_mapping_v2.csv"
            if not mapping_path.exists():
                self.logger.error(f"❌ 模块映射文件不存在: {mapping_path}")
                return False
            
            self.logger.info(f"✅ 聚类结果已存在: {mapping_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 检查聚类结果失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='内存高效的中介变量模块化分析')
    parser.add_argument('--variable_dict', type=str, 
                       default='all_nhanes_variables.csv',
                       help='变量字典文件路径')
    parser.add_argument('--data_dir', type=str,
                       default='NHANES_PROCESSED_CSV_merged_by_prefix',
                       help='数据文件目录')
    parser.add_argument('--output_dir', type=str,
                       default='mediator_analysis_results_v2',
                       help='输出目录')
    parser.add_argument('--max_missing_rate', type=float,
                       default=0.9,
                       help='最大允许缺失率 (默认: 0.9)')
    parser.add_argument('--min_variance', type=float,
                       default=1e-8,
                       help='最小方差阈值 (默认: 1e-8)')
    parser.add_argument('--n_clusters', type=int,
                       default=20,
                       help='聚类簇数 (默认: 20)')
    parser.add_argument('--linkage_method', type=str,
                       default='ward',
                       choices=['ward', 'complete', 'average', 'single'],
                       help='聚类链接方法 (默认: ward)')
    parser.add_argument('--max_memory_gb', type=float,
                       default=100.0,
                       help='最大内存使用限制(GB) (默认: 100.0)')
    parser.add_argument('--n_jobs', type=int,
                       default=-1,
                       help='并行作业数 (-1表示使用所有CPU核心)')
    parser.add_argument('--force_recalc_metadata', action='store_true',
                       help='强制重新计算元数据（即使已存在）')
    parser.add_argument('--start_from_stage', type=int,
                       default=1,
                       choices=[1, 2, 3, 4],
                       help='从哪个阶段开始运行: 1=元数据预计算, 2=相关性矩阵, 3=聚类分析, 4=模块代表生成')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = MemoryEfficientMediatorAnalyzer(
        variable_dict_path=args.variable_dict,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_memory_gb=args.max_memory_gb,
        n_jobs=args.n_jobs
    )
    
    # 运行分析
    success = analyzer.run_complete_analysis(
        max_missing_rate=args.max_missing_rate,
        min_variance=args.min_variance,
        n_clusters=args.n_clusters,
        linkage_method=args.linkage_method,
        force_recalculate_metadata=args.force_recalc_metadata,
        start_from_stage=args.start_from_stage
    )
    
    if success:
        print("\n🎉 分析成功完成!")
        sys.exit(0)
    else:
        print("\n❌ 分析失败!")
        sys.exit(1)

if __name__ == "__main__":
    main() 