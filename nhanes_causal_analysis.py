#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHANES 因果推断自动化分析系统
基于暴露→中介→结局的因果链进行系统性分析

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NHANESCausalAnalyzer:
    def __init__(self, base_data_path="Nhanes_processed_csv_merged", 
                 variables_file="all_nhanes_variables.csv",
                 results_base_dir="causal_analysis_results"):
        """
        初始化NHANES因果分析器
        
        参数:
        - base_data_path: 数据文件夹路径
        - variables_file: 变量索引文件
        - results_base_dir: 结果输出目录
        """
        self.base_data_path = Path(base_data_path)
        self.variables_file = variables_file
        self.results_base_dir = Path(results_base_dir)
        
        # 创建结果目录
        self.results_base_dir.mkdir(exist_ok=True)
        
        # 创建成功图表收集目录
        self.successful_charts_dir = self.results_base_dir / "successful_visualizations"
        self.successful_charts_dir.mkdir(exist_ok=True)
        
        # 设置日志系统
        self.setup_logging()
        
        # 加载变量信息
        self.load_variables_info()
        
        # 加载基准数据的RespondentSequenceNumber
        self.load_base_respondents()
        
        # 初始化结果记录
        self.analysis_results = []
        self.current_analysis_id = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.skipped_ab_invalid = 0  # 因A-B无重合或无变异性而跳过的分析数
        self.skipped_data_invalid = 0  # 因变量数据无效而跳过的分析数
        
    def setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = self.results_base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 设置主日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"nhanes_analysis_{timestamp}.log"
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
        self.logger.info("=" * 60)
        self.logger.info("NHANES因果推断自动化分析系统启动")
        self.logger.info("=" * 60)
        
    def clean_variable_name(self, name):
        """清理变量名，用于文件夹命名"""
        if pd.isna(name) or name is None:
            return "Unknown"
        
        # 转换为字符串并清理
        clean_name = str(name)
        # 移除或替换特殊字符
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)
        clean_name = re.sub(r'\s+', '_', clean_name)
        # 限制长度
        if len(clean_name) > 50:
            clean_name = clean_name[:50]
        return clean_name
        
    def is_valid_analysis_variable(self, variable_name):
        """判断变量是否适合用于分析"""
        if pd.isna(variable_name) or variable_name is None:
            return False
            
        variable_name = str(variable_name).lower()
        
        # 排除不适合分析的变量
        excluded_patterns = [
            'respondentsequencenumber',
            'sequence',
            'id',
            'weight',
            'comment',
            'status',
            'flag',
            'code',
            'cycle',
            'sample',
            'variance',
            'exam',
            'component',
            'yearrange',
            'interview',
            'stratum',
            'psu'
        ]
        
        for pattern in excluded_patterns:
            if pattern in variable_name:
                return False
                
        return True
        
    def load_variables_info(self):
        """加载变量信息表"""
        self.logger.info("正在加载变量信息...")
        self.variables_df = pd.read_csv(self.variables_file)
        
        # 按variable_index排序
        self.variables_df = self.variables_df.sort_values('variable_index').reset_index(drop=True)
        
        # 过滤有效的分析变量
        valid_mask = self.variables_df['variable_name'].apply(self.is_valid_analysis_variable)
        self.variables_df = self.variables_df[valid_mask].copy()
        
        # 获取各类别变量
        self.exposure_vars = self.variables_df[self.variables_df['category'] == 'A'].copy()
        self.mediator_vars = self.variables_df[self.variables_df['category'].isin(['B', 'B&C'])].copy()
        self.outcome_vars = self.variables_df[self.variables_df['category'].isin(['C', 'B&C'])].copy()
        
        self.logger.info(f"找到有效变量总数: {len(self.variables_df)}")
        self.logger.info(f"暴露变量(A): {len(self.exposure_vars)}")
        self.logger.info(f"中介变量(B/B&C): {len(self.mediator_vars)}")
        self.logger.info(f"结局变量(C/B&C): {len(self.outcome_vars)}")
        
        if len(self.exposure_vars) == 0 or len(self.mediator_vars) == 0 or len(self.outcome_vars) == 0:
            self.logger.error("错误: 某些类别的变量数量为0，无法进行分析")
            raise ValueError("变量类别不完整")
        
    def load_base_respondents(self):
        """加载基准数据的RespondentSequenceNumber"""
        base_file = self.base_data_path / "0_人口统计学信息_merged.csv"
        self.logger.info(f"正在加载基准数据: {base_file}")
        
        try:
            # 读取完整的基准数据RespondentSequenceNumber列
            self.logger.info("读取完整基准数据（可能需要较长时间）...")
            base_df = pd.read_csv(base_file, usecols=['RespondentSequenceNumber'])
            
            # 去除缺失值并转换为集合
            respondent_ids = base_df['RespondentSequenceNumber'].dropna().tolist()
            self.base_respondents = set(respondent_ids)
            
            # 记录RespondentSequenceNumber的范围
            min_id = min(self.base_respondents)
            max_id = max(self.base_respondents)
            
            self.logger.info(f"基准样本数量: {len(self.base_respondents)}")
            self.logger.info(f"RespondentSequenceNumber范围: {min_id} - {max_id}")
            
        except Exception as e:
            self.logger.error(f"加载基准数据失败: {e}")
            raise
        
    def load_variable_data(self, file_name, variable_name, filter_base=True):
        """
        加载指定变量的数据，并进行标准化处理
        
        参数:
        - file_name: 文件名
        - variable_name: 变量名
        - filter_base: 是否使用基准样本过滤
        
        返回:
        - DataFrame包含RespondentSequenceNumber和变量数据
        """
        file_path = self.base_data_path / file_name
        
        if not file_path.exists():
            self.logger.warning(f"文件不存在: {file_path}")
            return None
            
        try:
            # 只加载需要的列
            columns_to_load = ['RespondentSequenceNumber', variable_name]
            df = pd.read_csv(file_path, usecols=columns_to_load)
            
            self.logger.debug(f"从 {file_name} 加载变量 {variable_name}，原始数据量: {len(df)}")
            
            # 步骤1: 去除缺失值
            df = df.dropna()
            if len(df) == 0:
                return None
            
            # 步骤2: 检查并处理重复的RespondentSequenceNumber
            duplicates_count = df['RespondentSequenceNumber'].duplicated().sum()
            if duplicates_count > 0:
                df = df.drop_duplicates(subset=['RespondentSequenceNumber'], keep='first')
            
            # 步骤3: 按RespondentSequenceNumber排序
            df = df.sort_values('RespondentSequenceNumber').reset_index(drop=True)
            
            # 可选的基准样本过滤
            if filter_base and hasattr(self, 'base_respondents'):
                original_len = len(df)
                df = df[df['RespondentSequenceNumber'].isin(self.base_respondents)]
                self.logger.debug(f"基准样本过滤后数据量: {len(df)} (原始: {original_len})")
            
            # 步骤4: 检查变量数据质量
            variable_data = df[variable_name]
            
            # 处理包含长字符串的数据（可能是数据错误）
            if variable_data.dtype == 'object':
                # 检查是否有异常长的字符串
                max_length = variable_data.astype(str).str.len().max()
                if max_length > 100:  # 如果有超过100字符的字符串，可能是数据错误
                    self.logger.warning(f"变量 {variable_name} 包含异常长的字符串数据，最大长度: {max_length}")
                    # 过滤掉异常长的字符串
                    mask = variable_data.astype(str).str.len() <= 100
                    df = df[mask]
                    self.logger.debug(f"过滤异常数据后数据量: {len(df)}")
            
            # 简化日志：只记录基本信息
            if len(df) == 0:
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据时出错 {file_path}: {e}")
            return None
    
    def determine_variable_type(self, data):
        """
        判断变量类型：数值型或分类型
        
        参数:
        - data: pandas Series
        
        返回:
        - 'numerical' 或 'categorical'
        """
        # 检查数据类型
        if pd.api.types.is_numeric_dtype(data):
            # 数值型，但检查是否为类别编码
            unique_values = data.nunique()
            if unique_values <= 10 and data.min() >= 0 and data.max() <= 10:
                # 可能是编码的分类变量
                return 'categorical'
            else:
                return 'numerical'
        else:
            return 'categorical'
    
    def preprocess_variable(self, data, var_type):
        """
        预处理变量数据
        
        参数:
        - data: pandas Series
        - var_type: 'numerical' 或 'categorical'
        
        返回:
        - 预处理后的数据
        """
        try:
            if var_type == 'numerical':
                # 数值型变量：标准化
                # 检查是否包含无穷大或非常大的值
                finite_mask = np.isfinite(data.astype(float))
                if not finite_mask.all():
                    self.logger.warning(f"数值变量包含非有限值，过滤前数量: {len(data)}, 过滤后: {finite_mask.sum()}")
                    data = data[finite_mask]
                
                scaler = StandardScaler()
                processed_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
                self.logger.debug(f"数值变量标准化完成，均值: {processed_data.mean():.4f}, 标准差: {processed_data.std():.4f}")
                return processed_data
            else:
                # 分类型变量：标签编码
                # 检查类别数量
                unique_values = data.nunique()
                if unique_values == 1:
                    self.logger.warning(f"分类变量只有1个类别: {data.unique()}")
                    return None  # 返回None表示不适合分析
                elif unique_values > 20:
                    self.logger.warning(f"分类变量类别过多: {unique_values}，可能不适合分析")
                
                le = LabelEncoder()
                processed_data = le.fit_transform(data.astype(str))
                self.logger.debug(f"分类变量编码完成，类别数: {unique_values}, 编码范围: {processed_data.min()}-{processed_data.max()}")
                return processed_data
                
        except Exception as e:
            self.logger.error(f"变量预处理出错: {e}")
            return None
    
    def calculate_correlations(self, exposure_data, mediator_data, outcome_data):
        """
        计算变量间的相关性
        
        返回:
        - 相关性矩阵和显著性检验结果
        """
        data_matrix = np.column_stack([exposure_data, mediator_data, outcome_data])
        
        # 计算Pearson相关系数
        corr_matrix = np.corrcoef(data_matrix.T)
        
        # 计算显著性
        n = len(exposure_data)
        p_values = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    corr = corr_matrix[i, j]
                    t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                    p_values[i, j] = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
        
        return corr_matrix, p_values
    
    def mediation_analysis(self, exposure, mediator, outcome, 
                          exposure_type, mediator_type, outcome_type):
        """
        进行中介分析
        
        返回:
        - 中介分析结果字典
        """
        results = {}
        
        try:
            # 路径c：总效应 (X -> Y)
            if outcome_type == 'numerical':
                model_c = LinearRegression()
                model_c.fit(exposure.reshape(-1, 1), outcome)
                total_effect = model_c.coef_[0]
                results['total_effect'] = total_effect
            else:
                model_c = LogisticRegression()
                model_c.fit(exposure.reshape(-1, 1), outcome)
                total_effect = model_c.coef_[0][0]
                results['total_effect'] = total_effect
            
            # 路径a：X -> M
            if mediator_type == 'numerical':
                model_a = LinearRegression()
                model_a.fit(exposure.reshape(-1, 1), mediator)
                path_a = model_a.coef_[0]
                results['path_a'] = path_a
            else:
                model_a = LogisticRegression()
                model_a.fit(exposure.reshape(-1, 1), mediator)
                path_a = model_a.coef_[0][0]
                results['path_a'] = path_a
            
            # 路径b和c'：控制M时的X -> Y
            X_M = np.column_stack([exposure, mediator])
            
            if outcome_type == 'numerical':
                model_b = LinearRegression()
                model_b.fit(X_M, outcome)
                path_b = model_b.coef_[1]  # M -> Y
                direct_effect = model_b.coef_[0]  # X -> Y (控制M)
                results['path_b'] = path_b
                results['direct_effect'] = direct_effect
            else:
                model_b = LogisticRegression()
                model_b.fit(X_M, outcome)
                path_b = model_b.coef_[0][1]  # M -> Y
                direct_effect = model_b.coef_[0][0]  # X -> Y (控制M)
                results['path_b'] = path_b
                results['direct_effect'] = direct_effect
            
            # 间接效应
            indirect_effect = path_a * path_b
            results['indirect_effect'] = indirect_effect
            
            # 中介效应比例
            if abs(total_effect) > 1e-10:
                mediation_ratio = indirect_effect / total_effect
                results['mediation_ratio'] = mediation_ratio
            else:
                results['mediation_ratio'] = 0
            
        except Exception as e:
            print(f"中介分析出错: {e}")
            results = {
                'total_effect': 0,
                'path_a': 0,
                'path_b': 0,
                'direct_effect': 0,
                'indirect_effect': 0,
                'mediation_ratio': 0
            }
        
        return results
    
    def create_visualization(self, exposure_data, mediator_data, outcome_data,
                           exposure_name, mediator_name, outcome_name,
                           mediation_results, save_path):
        """
        创建可视化图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'因果分析: {exposure_name} → {mediator_name} → {outcome_name}', 
                     fontsize=16, fontweight='bold')
        
        # 1. 相关性热图
        data_df = pd.DataFrame({
            'Exposure': exposure_data,
            'Mediator': mediator_data,
            'Outcome': outcome_data
        })
        
        corr_matrix = data_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0, 0], vmin=-1, vmax=1)
        axes[0, 0].set_title('变量相关性矩阵')
        
        # 2. 散点图 - 暴露 vs 中介
        axes[0, 1].scatter(exposure_data, mediator_data, alpha=0.6)
        axes[0, 1].set_xlabel('暴露变量')
        axes[0, 1].set_ylabel('中介变量')
        axes[0, 1].set_title(f'{exposure_name} vs {mediator_name}')
        
        # 3. 散点图 - 中介 vs 结局
        axes[1, 0].scatter(mediator_data, outcome_data, alpha=0.6)
        axes[1, 0].set_xlabel('中介变量')
        axes[1, 0].set_ylabel('结局变量')
        axes[1, 0].set_title(f'{mediator_name} vs {outcome_name}')
        
        # 4. 中介效应图
        effects = [
            mediation_results['total_effect'],
            mediation_results['direct_effect'],
            mediation_results['indirect_effect']
        ]
        effect_names = ['总效应', '直接效应', '间接效应']
        
        bars = axes[1, 1].bar(effect_names, effects)
        axes[1, 1].set_title('中介效应分解')
        axes[1, 1].set_ylabel('效应大小')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 为条形图添加数值标签
        for bar, effect in zip(bars, effects):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{effect:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_trio(self, exposure_var, mediator_var, outcome_var):
        """
        分析一个A-B-C三元组
        
        返回:
        - 分析结果字典
        """
        self.current_analysis_id += 1
        
        # 创建分层文件夹结构：A → A-B → A-B-C
        exposure_name = self.clean_variable_name(exposure_var['variable_name'])
        mediator_name = self.clean_variable_name(mediator_var['variable_name'])
        outcome_name = self.clean_variable_name(outcome_var['variable_name'])
        
        # 层次1: A (暴露变量)
        a_folder_name = f"A_{exposure_name}"
        a_dir = self.results_base_dir / a_folder_name
        a_dir.mkdir(exist_ok=True)
        
        # 层次2: A-B (暴露-中介组合)
        ab_folder_name = f"A_{exposure_name}__B_{mediator_name}"
        ab_dir = a_dir / ab_folder_name
        ab_dir.mkdir(exist_ok=True)
        
        # 层次3: A-B-C (具体分析)
        abc_folder_name = f"A_{exposure_name}__B_{mediator_name}__C_{outcome_name}"
        analysis_id = f"analysis_{self.current_analysis_id:05d}_{abc_folder_name}"
        analysis_dir = ab_dir / analysis_id
        analysis_dir.mkdir(exist_ok=True)
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"开始分析 #{self.current_analysis_id}: {abc_folder_name}")
        self.logger.info(f"  路径: {a_folder_name}/{ab_folder_name}/{analysis_id}")
        self.logger.info(f"  暴露变量(A): {exposure_var['variable_name']} (来自 {exposure_var['file_name']})")
        self.logger.info(f"  中介变量(B): {mediator_var['variable_name']} (来自 {mediator_var['file_name']})")
        self.logger.info(f"  结局变量(C): {outcome_var['variable_name']} (来自 {outcome_var['file_name']})")
        
        # 创建当前分析的日志文件
        analysis_log_file = analysis_dir / "analysis.log"
        analysis_logger = logging.getLogger(f"analysis_{self.current_analysis_id}")
        analysis_logger.setLevel(logging.DEBUG)
        
        # 清除之前的处理器
        for handler in analysis_logger.handlers[:]:
            analysis_logger.removeHandler(handler)
            
        # 添加文件处理器
        file_handler = logging.FileHandler(analysis_log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        analysis_logger.addHandler(file_handler)
        
        analysis_logger.info(f"开始分析: {abc_folder_name}")
        analysis_logger.info(f"分析ID: {analysis_id}")
        analysis_logger.info(f"分析路径: {a_folder_name}/{ab_folder_name}/{analysis_id}")
        
        try:
            # 简化日志：加载A-B变量
            exposure_df = self.load_variable_data(exposure_var['file_name'], exposure_var['variable_name'], filter_base=False)
            mediator_df = self.load_variable_data(mediator_var['file_name'], mediator_var['variable_name'], filter_base=False)
            
            if any(df is None for df in [exposure_df, mediator_df]):
                analysis_logger.warning("跳过：A或B变量无效")
                self.skipped_data_invalid += 1
                return None
            
            # A-B重合检查已在外层循环完成，这里直接加载结局变量
            outcome_df = self.load_variable_data(outcome_var['file_name'], outcome_var['variable_name'], filter_base=False)
            
            if outcome_df is None:
                analysis_logger.warning("跳过：C变量无效")
                self.skipped_data_invalid += 1
                return None
            
            # 合并数据
            merged_data = exposure_df.merge(mediator_df, on='RespondentSequenceNumber', how='inner')
            merged_data = merged_data.merge(outcome_df, on='RespondentSequenceNumber', how='inner')
            analysis_logger.info(f"合并样本数: {len(merged_data)}")
            if len(merged_data) > 0:
                final_min = merged_data['RespondentSequenceNumber'].min()
                final_max = merged_data['RespondentSequenceNumber'].max()
                analysis_logger.info(f"最终合并后ID范围: {final_min:.0f} - {final_max:.0f}")
                self.logger.info(f"  共同ID范围: {final_min:.0f} - {final_max:.0f} (样本数: {len(merged_data)})")
            
            if len(merged_data) < 100:  # 样本量太小
                error_msg = f"样本量不足 ({len(merged_data)})"
                self.logger.warning(f"  跳过分析 #{self.current_analysis_id}: {error_msg}")
                analysis_logger.error(error_msg)
                self.failed_analyses += 1
                return None
            
            self.logger.info(f"  有效样本数: {len(merged_data)}")
            analysis_logger.info(f"有效样本数: {len(merged_data)}")
            
            # 提取变量数据
            exposure_data = merged_data[exposure_var['variable_name']]
            mediator_data = merged_data[mediator_var['variable_name']]
            outcome_data = merged_data[outcome_var['variable_name']]
            
            # 判断变量类型
            exposure_type = self.determine_variable_type(exposure_data)
            mediator_type = self.determine_variable_type(mediator_data)
            outcome_type = self.determine_variable_type(outcome_data)
            
            self.logger.info(f"  变量类型 - 暴露: {exposure_type}, 中介: {mediator_type}, 结局: {outcome_type}")
            analysis_logger.info(f"变量类型 - 暴露: {exposure_type}, 中介: {mediator_type}, 结局: {outcome_type}")
            
            # 预处理数据
            analysis_logger.info("开始预处理数据...")
            exposure_processed = self.preprocess_variable(exposure_data, exposure_type)
            mediator_processed = self.preprocess_variable(mediator_data, mediator_type)
            outcome_processed = self.preprocess_variable(outcome_data, outcome_type)
            
            # 检查预处理结果
            if any(x is None for x in [exposure_processed, mediator_processed, outcome_processed]):
                error_msg = "数据预处理失败，可能存在数据质量问题"
                self.logger.warning(f"  跳过分析 #{self.current_analysis_id}: {error_msg}")
                analysis_logger.error(error_msg)
                self.failed_analyses += 1
                return None
        
        except Exception as e:
            error_msg = f"分析过程发生异常: {str(e)}"
            self.logger.error(f"  分析 #{self.current_analysis_id} 失败: {error_msg}")
            analysis_logger.error(error_msg)
            self.failed_analyses += 1
            return None
        
        # 计算相关性
        analysis_logger.info("计算变量间相关性...")
        corr_matrix, p_values = self.calculate_correlations(
            exposure_processed, mediator_processed, outcome_processed)
        
        analysis_logger.info(f"相关性系数:")
        analysis_logger.info(f"  暴露-中介: {corr_matrix[0, 1]:.4f} (p={p_values[0, 1]:.4f})")
        analysis_logger.info(f"  中介-结局: {corr_matrix[1, 2]:.4f} (p={p_values[1, 2]:.4f})")
        analysis_logger.info(f"  暴露-结局: {corr_matrix[0, 2]:.4f} (p={p_values[0, 2]:.4f})")
        
        # 中介分析
        analysis_logger.info("进行中介分析...")
        mediation_results = self.mediation_analysis(
            exposure_processed, mediator_processed, outcome_processed,
            exposure_type, mediator_type, outcome_type)
        
        analysis_logger.info(f"中介分析结果:")
        analysis_logger.info(f"  总效应: {mediation_results['total_effect']:.4f}")
        analysis_logger.info(f"  直接效应: {mediation_results['direct_effect']:.4f}")
        analysis_logger.info(f"  间接效应: {mediation_results['indirect_effect']:.4f}")
        analysis_logger.info(f"  中介效应比例: {mediation_results['mediation_ratio']:.4f}")
        
        # 创建可视化
        analysis_logger.info("生成可视化图表...")
        viz_path = analysis_dir / "visualization.png"
        self.create_visualization(
            exposure_processed, mediator_processed, outcome_processed,
            exposure_var['variable_name'], mediator_var['variable_name'], 
            outcome_var['variable_name'], mediation_results, viz_path)
        analysis_logger.info(f"可视化图表已保存: {viz_path}")
        
        # 复制可视化图表到成功收集目录
        try:
            # 使用分层信息创建更有意义的文件名
            chart_name = f"{a_folder_name}__{ab_folder_name.split('__', 1)[1]}__{analysis_id.split('__')[-1]}_visualization.png"
            successful_chart_path = self.successful_charts_dir / chart_name
            shutil.copy2(viz_path, successful_chart_path)
            analysis_logger.info(f"可视化图表已复制到成功收集目录: {successful_chart_path}")
            self.logger.info(f"  ✓ 图表已收集: {chart_name}")
            self.logger.info(f"  ✓ 路径: {a_folder_name}/{ab_folder_name}/{analysis_id}")
        except Exception as e:
            analysis_logger.warning(f"复制图表到收集目录失败: {e}")
        
        # 保存详细结果
        analysis_logger.info("保存分析结果...")
        
        detailed_results = {
            'analysis_info': {
                'analysis_id': analysis_id,
                'folder_name': abc_folder_name,
                'full_path': f"{a_folder_name}/{ab_folder_name}/{analysis_id}",
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            },
            'variables': {
                'exposure': {
                    'name': exposure_var['variable_name'],
                    'file': exposure_var['file_name'],
                    'type': exposure_type,
                    'category': exposure_var['category'],
                    'index': exposure_var['variable_index']
                },
                'mediator': {
                    'name': mediator_var['variable_name'],
                    'file': mediator_var['file_name'],
                    'type': mediator_type,
                    'category': mediator_var['category'],
                    'index': mediator_var['variable_index']
                },
                'outcome': {
                    'name': outcome_var['variable_name'],
                    'file': outcome_var['file_name'],
                    'type': outcome_type,
                    'category': outcome_var['category'],
                    'index': outcome_var['variable_index']
                }
            },
            'sample_size': len(merged_data),
            'correlations': {
                'exposure_mediator': float(corr_matrix[0, 1]),
                'mediator_outcome': float(corr_matrix[1, 2]),
                'exposure_outcome': float(corr_matrix[0, 2])
            },
            'p_values': {
                'exposure_mediator': float(p_values[0, 1]),
                'mediator_outcome': float(p_values[1, 2]),
                'exposure_outcome': float(p_values[0, 2])
            },
            'mediation_analysis': mediation_results,
            'data_summary': {
                'exposure': {
                    'mean': float(exposure_data.mean()),
                    'std': float(exposure_data.std()),
                    'min': float(exposure_data.min()),
                    'max': float(exposure_data.max()),
                    'unique_count': int(exposure_data.nunique())
                },
                'mediator': {
                    'mean': float(mediator_data.mean()),
                    'std': float(mediator_data.std()),
                    'min': float(mediator_data.min()),
                    'max': float(mediator_data.max()),
                    'unique_count': int(mediator_data.nunique())
                },
                'outcome': {
                    'mean': float(outcome_data.mean()),
                    'std': float(outcome_data.std()),
                    'min': float(outcome_data.min()),
                    'max': float(outcome_data.max()),
                    'unique_count': int(outcome_data.nunique())
                }
            },
            'file_paths': {
                'visualization': 'visualization.png',
                'data': 'data.csv',
                'log': 'analysis.log',
                'results': 'results.json'
            }
        }
        
        # 保存JSON结果
        results_file = analysis_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        analysis_logger.info(f"结果已保存: {results_file}")
        
        # 保存CSV数据
        data_file = analysis_dir / "data.csv"
        merged_data.to_csv(data_file, index=False, encoding='utf-8')
        analysis_logger.info(f"数据已保存: {data_file}")
        
        # 保存数据描述性统计
        stats_file = analysis_dir / "descriptive_stats.csv"
        stats_df = merged_data.describe(include='all')
        stats_df.to_csv(stats_file, encoding='utf-8')
        analysis_logger.info(f"描述性统计已保存: {stats_file}")
        
        # 添加到总结果
        self.analysis_results.append(detailed_results)
        self.successful_analyses += 1
        
        self.logger.info(f"  ✓ 完成分析 #{self.current_analysis_id}: {abc_folder_name}")
        analysis_logger.info("分析完成！")
        
        # 关闭分析日志处理器
        for handler in analysis_logger.handlers[:]:
            handler.close()
            analysis_logger.removeHandler(handler)
        
        return detailed_results
    
    def run_full_analysis(self, max_analyses=None):
        """
        运行完整的因果分析
        
        参数:
        - max_analyses: 最大分析数量（用于测试）
        """
        start_time = datetime.now()
        self.logger.info("开始全面因果分析...")
        
        total_possible = len(self.exposure_vars) * len(self.mediator_vars) * len(self.outcome_vars)
        self.logger.info(f"计划分析组合数: {len(self.exposure_vars)} × {len(self.mediator_vars)} × {len(self.outcome_vars)} = {total_possible}")
        
        if max_analyses:
            self.logger.info(f"限制最大分析数量: {max_analyses}")
        
        total_combinations = 0
        
        for i, exposure_var in self.exposure_vars.iterrows():
            self.logger.info(f"\n处理暴露变量 {i+1}/{len(self.exposure_vars)}: {exposure_var['variable_name']}")
            
            for j, mediator_var in self.mediator_vars.iterrows():
                # 确保中介变量和暴露变量不同
                if mediator_var['variable_name'] == exposure_var['variable_name']:
                    continue
                
                # 提前检查A-B组合是否有重合的RespondentSequenceNumber
                self.logger.info(f"  检查A-B组合: {exposure_var['variable_name']} vs {mediator_var['variable_name']}")
                
                try:
                    # 加载A和B变量数据来检查ID重合
                    exposure_df = self.load_variable_data(exposure_var['file_name'], exposure_var['variable_name'], filter_base=False)
                    mediator_df = self.load_variable_data(mediator_var['file_name'], mediator_var['variable_name'], filter_base=False)
                    
                    if exposure_df is None or mediator_df is None:
                        self.logger.warning(f"  ⚠️ A-B数据加载失败，跳过整个A-B组合")
                        continue
                    
                    # 检查ID重合
                    exposure_ids = set(exposure_df['RespondentSequenceNumber'])
                    mediator_ids = set(mediator_df['RespondentSequenceNumber'])
                    ab_intersection = exposure_ids & mediator_ids
                    
                    self.logger.info(f"  A变量ID数: {len(exposure_ids)}, B变量ID数: {len(mediator_ids)}")
                    self.logger.info(f"  A-B交集数量: {len(ab_intersection)}")
                    
                    if len(ab_intersection) == 0:
                        self.logger.info(f"  ⚠️ A-B无共同样本，跳过整个A-B组合（节省所有C变量检查）")
                        self.skipped_ab_invalid += len(self.outcome_vars)  # 统计跳过的A-B-C组合数
                        continue
                    
                    if len(ab_intersection) < 100:
                        self.logger.info(f"  ⚠️ A-B共同样本太少 ({len(ab_intersection)})，跳过整个A-B组合")
                        self.skipped_ab_invalid += len(self.outcome_vars)
                        continue
                    
                    # 进一步检查A-B合并后的变异性
                    ab_merged = exposure_df.merge(mediator_df, on='RespondentSequenceNumber', how='inner')
                    if len(ab_merged) == 0:
                        self.logger.info(f"  ⚠️ A-B合并后无数据，跳过整个A-B组合")
                        self.skipped_ab_invalid += len(self.outcome_vars)
                        continue
                    
                    # 检查变量变异性（这是关键优化！）
                    exposure_var_name = exposure_var['variable_name']
                    mediator_var_name = mediator_var['variable_name']
                    
                    exposure_unique = ab_merged[exposure_var_name].nunique()
                    mediator_unique = ab_merged[mediator_var_name].nunique()
                    
                    self.logger.info(f"  变异性检查: A变量 {exposure_unique} 个不同值, B变量 {mediator_unique} 个不同值")
                    
                    if exposure_unique < 2:
                        self.logger.info(f"  ❌ A变量无变异性 (只有 {exposure_unique} 个值), 跳过整个A-B组合")
                        self.skipped_ab_invalid += len(self.outcome_vars)
                        continue
                    
                    if mediator_unique < 2:
                        self.logger.info(f"  ❌ B变量无变异性 (只有 {mediator_unique} 个值), 跳过整个A-B组合")
                        self.skipped_ab_invalid += len(self.outcome_vars)
                        continue
                    
                    self.logger.info(f"  ✅ A-B组合有效 ({len(ab_intersection)} 共同样本，变异性充足)，开始检查所有C变量")
                    
                    # 创建A-B组合的文件夹结构（为后续分析做准备）
                    exposure_name_clean = self.clean_variable_name(exposure_var['variable_name'])
                    mediator_name_clean = self.clean_variable_name(mediator_var['variable_name'])
                    a_folder_name_prep = f"A_{exposure_name_clean}"
                    ab_folder_name_prep = f"A_{exposure_name_clean}__B_{mediator_name_clean}"
                    
                    a_dir_prep = self.results_base_dir / a_folder_name_prep
                    ab_dir_prep = a_dir_prep / ab_folder_name_prep
                    a_dir_prep.mkdir(exist_ok=True)
                    ab_dir_prep.mkdir(exist_ok=True)
                    
                    self.logger.info(f"  📁 已准备A-B组合目录: {a_folder_name_prep}/{ab_folder_name_prep}")
                    
                except Exception as e:
                    self.logger.error(f"  A-B预检查出错: {e}，跳过整个A-B组合")
                    continue
                
                # 只有A-B有效重合时，才进行所有C变量的分析
                for k, outcome_var in self.outcome_vars.iterrows():
                    # 确保结局变量和前两个变量不同
                    if (outcome_var['variable_name'] == exposure_var['variable_name'] or
                        outcome_var['variable_name'] == mediator_var['variable_name']):
                        continue
                    
                    total_combinations += 1
                    
                    if max_analyses and self.successful_analyses >= max_analyses:
                        self.logger.info(f"达到最大分析数量限制 {max_analyses}")
                        break
                    
                    try:
                        result = self.analyze_trio(exposure_var, mediator_var, outcome_var)
                        
                        # 每10个分析输出进度
                        if total_combinations % 10 == 0:
                            success_rate = (self.successful_analyses / total_combinations) * 100
                            self.logger.info(f"进度: {total_combinations} 组合完成，成功率: {success_rate:.1f}%")
                            
                    except Exception as e:
                        self.logger.error(f"分析出错: {e}")
                        self.failed_analyses += 1
                        continue
                
                if max_analyses and self.successful_analyses >= max_analyses:
                    break
            
            if max_analyses and self.successful_analyses >= max_analyses:
                break
        
        # 分析完成统计
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info("分析完成！")
        self.logger.info("="*60)
        self.logger.info(f"总分析组合数: {total_combinations}")
        self.logger.info(f"成功分析数: {self.successful_analyses}")
        self.logger.info(f"失败分析数: {self.failed_analyses}")
        self.logger.info(f"A-B无效跳过数: {self.skipped_ab_invalid}")
        self.logger.info(f"数据无效跳过数: {self.skipped_data_invalid}")
        total_skipped = self.skipped_ab_invalid + self.skipped_data_invalid
        self.logger.info(f"成功率: {(self.successful_analyses/total_combinations)*100:.1f}%")
        self.logger.info(f"总跳过率: {(total_skipped/total_combinations)*100:.1f}%")
        self.logger.info(f"分析耗时: {duration}")
        self.logger.info(f"优化效果: 跳过了 {total_skipped} 个无效组合，节省了计算时间")
        
        # 生成总结报告
        if self.successful_analyses > 0:
            self.generate_summary_report()
        else:
            self.logger.warning("没有成功的分析，跳过生成总结报告")
    
    def generate_summary_report(self):
        """生成总结报告"""
        self.logger.info("生成总结报告...")
        
        summary_dir = self.results_base_dir / "summary"
        summary_dir.mkdir(exist_ok=True)
        
        if not self.analysis_results:
            self.logger.warning("没有分析结果可供汇总")
            return
        
        # 创建总结DataFrame
        summary_data = []
        for result in self.analysis_results:
            summary_data.append({
                'analysis_id': result['analysis_info']['analysis_id'],
                'folder_name': result['analysis_info']['folder_name'],
                'exposure_name': result['variables']['exposure']['name'],
                'mediator_name': result['variables']['mediator']['name'],
                'outcome_name': result['variables']['outcome']['name'],
                'exposure_type': result['variables']['exposure']['type'],
                'mediator_type': result['variables']['mediator']['type'],
                'outcome_type': result['variables']['outcome']['type'],
                'sample_size': result['sample_size'],
                'exposure_mediator_corr': result['correlations']['exposure_mediator'],
                'mediator_outcome_corr': result['correlations']['mediator_outcome'],
                'exposure_outcome_corr': result['correlations']['exposure_outcome'],
                'total_effect': result['mediation_analysis']['total_effect'],
                'direct_effect': result['mediation_analysis']['direct_effect'],
                'indirect_effect': result['mediation_analysis']['indirect_effect'],
                'mediation_ratio': result['mediation_analysis']['mediation_ratio'],
                'timestamp': result['analysis_info']['timestamp']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = summary_dir / "analysis_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        self.logger.info(f"分析汇总表已保存: {summary_file}")
        
        # 保存完整结果JSON
        all_results_file = summary_dir / "all_results.json"
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"完整结果已保存: {all_results_file}")
        
        # 创建分析统计报告
        total_attempts = self.successful_analyses + self.failed_analyses + self.skipped_ab_invalid + self.skipped_data_invalid
        total_skipped = self.skipped_ab_invalid + self.skipped_data_invalid
        stats_report = {
            'analysis_statistics': {
                'total_successful': self.successful_analyses,
                'total_failed': self.failed_analyses,
                'total_skipped_ab_invalid': self.skipped_ab_invalid,
                'total_skipped_data_invalid': self.skipped_data_invalid,
                'total_attempts': total_attempts,
                'success_rate': (self.successful_analyses / total_attempts) * 100 if total_attempts > 0 else 0,
                'skip_rate': (total_skipped / total_attempts) * 100 if total_attempts > 0 else 0,
                'optimization_benefit': f"Saved {total_skipped} analysis attempts (overlap check + integer filter)",
                'total_variables_used': {
                    'exposure': len(self.exposure_vars),
                    'mediator': len(self.mediator_vars),
                    'outcome': len(self.outcome_vars)
                }
            },
            'data_quality_summary': {
                'average_sample_size': float(summary_df['sample_size'].mean()),
                'median_sample_size': float(summary_df['sample_size'].median()),
                'min_sample_size': int(summary_df['sample_size'].min()),
                'max_sample_size': int(summary_df['sample_size'].max())
            },
            'effect_size_summary': {
                'significant_mediation_count': len(summary_df[abs(summary_df['mediation_ratio']) > 0.1]),
                'strong_mediation_count': len(summary_df[abs(summary_df['mediation_ratio']) > 0.3]),
                'average_mediation_ratio': float(summary_df['mediation_ratio'].mean())
            }
        }
        
        stats_file = summary_dir / "analysis_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"分析统计报告已保存: {stats_file}")
        
        # 创建总结可视化
        self.create_summary_visualization(summary_df, summary_dir)
        
        # 创建成功图表索引
        self.create_successful_charts_index()
        
        # 生成汇总数据表（前10000条）
        self.create_summary_table(summary_dir, max_rows=10000)
        
        self.logger.info(f"总结报告已保存到: {summary_dir}")
        self.logger.info(f"生成了 {len(self.analysis_results)} 个成功分析的详细报告")
        self.logger.info(f"成功图表收集在: {self.successful_charts_dir}")
        
        # 统计成功图表数量
        chart_files = list(self.successful_charts_dir.glob("*.png"))
        self.logger.info(f"共收集了 {len(chart_files)} 个可视化图表")
    
    def create_summary_visualization(self, summary_df, save_dir):
        """创建总结可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NHANES因果分析总结', fontsize=16, fontweight='bold')
        
        # 1. 中介效应比例分布
        axes[0, 0].hist(summary_df['mediation_ratio'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('中介效应比例')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('中介效应比例分布')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 2. 样本量分布
        axes[0, 1].hist(summary_df['sample_size'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('样本量')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('样本量分布')
        
        # 3. 效应大小散点图
        axes[1, 0].scatter(summary_df['total_effect'], summary_df['indirect_effect'], alpha=0.6)
        axes[1, 0].set_xlabel('总效应')
        axes[1, 0].set_ylabel('间接效应')
        axes[1, 0].set_title('总效应 vs 间接效应')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 4. 相关性热图
        corr_cols = ['exposure_mediator_corr', 'mediator_outcome_corr', 'exposure_outcome_corr']
        corr_matrix = summary_df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[1, 1], vmin=-1, vmax=1)
        axes[1, 1].set_title('相关性模式的相关性')
        
        plt.tight_layout()
        plt.savefig(save_dir / "summary_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self, summary_dir, max_rows=10000):
        """创建汇总数据表"""
        if not self.analysis_results:
            return
            
        self.logger.info("创建汇总数据表...")
        
        # 准备汇总数据
        summary_data = []
        for i, result in enumerate(self.analysis_results[:max_rows]):
            try:
                row = {
                    'Analysis_ID': result['analysis_info']['analysis_id'],
                    'A_Variable': result['variables']['exposure']['name'],
                    'B_Variable': result['variables']['mediator']['name'], 
                    'C_Variable': result['variables']['outcome']['name'],
                    'A_File': result['variables']['exposure']['file_name'],
                    'B_File': result['variables']['mediator']['file_name'],
                    'C_File': result['variables']['outcome']['file_name'],
                    'Sample_Size': result['sample_size'],
                    'A_Type': result['variables']['exposure']['type'],
                    'B_Type': result['variables']['mediator']['type'],
                    'C_Type': result['variables']['outcome']['type'],
                    'Total_Effect': result['mediation_analysis']['total_effect'],
                    'Direct_Effect': result['mediation_analysis']['direct_effect'],
                    'Indirect_Effect': result['mediation_analysis']['indirect_effect'],
                    'Mediation_Ratio': result['mediation_analysis']['mediation_ratio'],
                    'A_B_Correlation': result['correlations']['exposure_mediator']['coefficient'],
                    'B_C_Correlation': result['correlations']['mediator_outcome']['coefficient'], 
                    'A_C_Correlation': result['correlations']['exposure_outcome']['coefficient'],
                    'A_B_P_Value': result['correlations']['exposure_mediator']['p_value'],
                    'B_C_P_Value': result['correlations']['mediator_outcome']['p_value'],
                    'A_C_P_Value': result['correlations']['exposure_outcome']['p_value'],
                    'Timestamp': result['analysis_info']['timestamp'][:19],  # 去掉毫秒
                    'Status': result['analysis_info']['status']
                }
                summary_data.append(row)
            except Exception as e:
                self.logger.warning(f"处理结果 {i} 时出错: {e}")
                continue
        
        if summary_data:
            # 创建DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # 保存为CSV
            csv_path = summary_dir / f"summary_table_{len(summary_data)}_results.csv"
            summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # 保存为Excel
            excel_path = summary_dir / f"summary_table_{len(summary_data)}_results.xlsx"
            summary_df.to_excel(excel_path, index=False)
            
            self.logger.info(f"汇总数据表已保存: {csv_path}")
            self.logger.info(f"汇总数据表已保存: {excel_path}")
            self.logger.info(f"数据表包含 {len(summary_data)} 行，{len(summary_df.columns)} 列")
        else:
            self.logger.warning("无有效数据生成汇总表")
    
    def create_successful_charts_index(self):
        """创建成功图表的索引文件"""
        self.logger.info("创建成功图表索引...")
        
        chart_index = []
        chart_files = list(self.successful_charts_dir.glob("*.png"))
        
        for chart_file in sorted(chart_files):
            # 从文件名中提取信息（新的分层命名格式）
            file_name = chart_file.name
            if file_name.endswith("_visualization.png"):
                # 新格式: A_exposure__B_mediator__C_outcome_visualization.png
                # 需要重新构造analysis_id来匹配
                base_name = file_name.replace("_visualization.png", "")
                
                # 查找对应的分析结果（通过folder_name匹配）
                matching_result = None
                for result in self.analysis_results:
                    # 检查文件名是否包含分析的folder_name信息
                    if (result['analysis_info']['folder_name'] in base_name or 
                        base_name in result['analysis_info'].get('full_path', '')):
                        matching_result = result
                        break
                
                if matching_result:
                    chart_info = {
                        'chart_file': file_name,
                        'analysis_id': matching_result['analysis_info']['analysis_id'],
                        'folder_name': matching_result['analysis_info']['folder_name'],
                        'full_path': matching_result['analysis_info'].get('full_path', ''),
                        'exposure_variable': matching_result['variables']['exposure']['name'],
                        'mediator_variable': matching_result['variables']['mediator']['name'],
                        'outcome_variable': matching_result['variables']['outcome']['name'],
                        'sample_size': matching_result['sample_size'],
                        'mediation_ratio': matching_result['mediation_analysis']['mediation_ratio'],
                        'total_effect': matching_result['mediation_analysis']['total_effect'],
                        'indirect_effect': matching_result['mediation_analysis']['indirect_effect'],
                        'timestamp': matching_result['analysis_info']['timestamp']
                    }
                    chart_index.append(chart_info)
        
        # 保存图表索引CSV
        if chart_index:
            chart_index_df = pd.DataFrame(chart_index)
            index_file = self.successful_charts_dir / "charts_index.csv"
            chart_index_df.to_csv(index_file, index=False, encoding='utf-8')
            self.logger.info(f"图表索引已保存: {index_file}")
            
            # 保存JSON格式的详细索引
            index_json_file = self.successful_charts_dir / "charts_index.json"
            with open(index_json_file, 'w', encoding='utf-8') as f:
                json.dump(chart_index, f, ensure_ascii=False, indent=2)
            self.logger.info(f"图表详细索引已保存: {index_json_file}")
            
            # 创建图表概览HTML
            self.create_charts_overview_html(chart_index)
        else:
            self.logger.warning("没有找到成功的图表文件")
    
    def create_charts_overview_html(self, chart_index):
        """创建图表概览的HTML文件"""
        try:
            html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NHANES因果分析图表概览</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-item { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #fafafa; }
        .chart-item img { width: 100%; height: auto; border-radius: 5px; margin-bottom: 10px; }
        .chart-info h3 { margin: 0 0 10px 0; color: #2c3e50; font-size: 16px; }
        .chart-info p { margin: 5px 0; font-size: 14px; color: #555; }
        .effect-value { font-weight: bold; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .stats { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .stats h2 { margin-top: 0; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NHANES因果分析图表概览</h1>
        
        <div class="stats">
            <h2>分析统计</h2>
            <p><strong>成功分析数量:</strong> {total_charts}</p>
            <p><strong>生成时间:</strong> {generation_time}</p>
        </div>
        
        <div class="chart-grid">
""".format(
                total_charts=len(chart_index),
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            for chart in chart_index:
                effect_class = "positive" if chart['mediation_ratio'] > 0 else "negative"
                
                html_content += f"""
            <div class="chart-item">
                <img src="{chart['chart_file']}" alt="分析图表">
                <div class="chart-info">
                    <h3>{chart['folder_name']}</h3>
                    <p><strong>文件路径:</strong> <small>{chart.get('full_path', 'N/A')}</small></p>
                    <p><strong>暴露变量:</strong> {chart['exposure_variable']}</p>
                    <p><strong>中介变量:</strong> {chart['mediator_variable']}</p>
                    <p><strong>结局变量:</strong> {chart['outcome_variable']}</p>
                    <p><strong>样本量:</strong> {chart['sample_size']}</p>
                    <p><strong>中介效应比例:</strong> <span class="effect-value {effect_class}">{chart['mediation_ratio']:.4f}</span></p>
                    <p><strong>总效应:</strong> <span class="effect-value">{chart['total_effect']:.4f}</span></p>
                    <p><strong>间接效应:</strong> <span class="effect-value">{chart['indirect_effect']:.4f}</span></p>
                    <p><strong>分析ID:</strong> {chart['analysis_id']}</p>
                </div>
            </div>
"""
            
            html_content += """
        </div>
    </div>
</body>
</html>
"""
            
            html_file = self.successful_charts_dir / "charts_overview.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"图表概览HTML已创建: {html_file}")
            
        except Exception as e:
            self.logger.error(f"创建HTML概览失败: {e}")

def main():
    """主函数"""
    print("NHANES因果推断自动化分析系统")
    print("=" * 50)
    
    try:
        # 初始化分析器
        analyzer = NHANESCausalAnalyzer()
        
        # 运行分析（限制前10000个组合）
        analyzer.run_full_analysis(max_analyses=10000)
        
        analyzer.logger.info("\n" + "="*60)
        analyzer.logger.info("分析系统运行完成！")
        analyzer.logger.info(f"结果保存在 {analyzer.results_base_dir} 目录中")
        analyzer.logger.info(f"日志文件: {analyzer.log_file}")
        
        # 输出成功图表收集信息
        chart_files = list(analyzer.successful_charts_dir.glob("*.png"))
        if chart_files:
            analyzer.logger.info(f"成功生成图表数量: {len(chart_files)}")
            analyzer.logger.info(f"图表收集目录: {analyzer.successful_charts_dir}")
            analyzer.logger.info(f"图表概览页面: {analyzer.successful_charts_dir / 'charts_overview.html'}")
        else:
            analyzer.logger.info("未生成成功的图表")
        
        analyzer.logger.info("="*60)
        
    except Exception as e:
        print(f"系统运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 