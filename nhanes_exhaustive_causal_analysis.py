#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHANES 全变量穷举因果路径分析脚本
==========================================

此脚本对所有NHANES变量进行全排列组合分析，系统性地发现：
- 3变量因果链：A → B → C
- 4变量因果链：A → B → C → D  
- 5变量因果链：A → B → C → D → E

不预设变量角色，每个变量都可能作为暴露、中介或结局。
输出所有结果，不进行显著性筛选。

作者: AI Assistant
日期: 2024年
"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import logging
from itertools import permutations, combinations
import json
from datetime import datetime
import gc

# 设置matplotlib为英文环境，避免中文字体问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 设置详细日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nhanes_exhaustive_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
ALPHA = 0.05  # 统计显著性水平
DATA_DIR = "./NHANES_PROCESSED_CSV"
OUTPUT_DIR = "./output_exhaustive_analysis"
MIN_SAMPLE_SIZE = 100  # 最小样本量要求
MAX_CHAIN_LENGTH = 5   # 最大因果链长度

class NHANESExhaustiveAnalysis:
    """NHANES全变量穷举因果路径分析类"""
    
    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.master_df = None
        self.analysis_variables = []
        self.results = {
            '3_chain': [],
            '4_chain': [],
            '5_chain': []
        }
        self.summary_stats = {}
        
    def discover_all_merged_files(self):
        """Discover all merged data files"""
        logger.info("🔍 Discovering all merged data files...")
        
        merged_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('_merged.csv'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.data_dir)
                    merged_files.append(relative_path)
        
        logger.info(f"📁 Found {len(merged_files)} merged data files")
        return sorted(merged_files)
    
    def load_and_merge_data(self):
        """Load and merge all available data"""
        logger.info("="*60)
        logger.info("Starting data loading and merging...")
        logger.info("="*60)
        
        # Discover all merged files
        all_merged_files = self.discover_all_merged_files()
        
        # Find demographics file as base
        demo_file = None
        for file_path in all_merged_files:
            if '人口统计学信息' in str(file_path):
                demo_file = file_path
                break
        
        if demo_file is None:
            logger.error("❌ Demographics file not found")
            return False
        
        # Load demographics data as base
        demo_path = self.data_dir / demo_file
        logger.info(f"📁 Loading base demographics data: {demo_path}")
        
        try:
            demo_chunks = pd.read_csv(demo_path, chunksize=10000)
            self.master_df = pd.concat(demo_chunks, ignore_index=True)
            logger.info(f"✅ Demographics data loaded: {len(self.master_df):,} rows")
        except Exception as e:
            logger.error(f"❌ Failed to load demographics data: {e}")
            return False
        
        # Merge all other data files
        merge_keys = ['RespondentSequenceNumber', 'YearRange']
        successful_merges = 0
        failed_merges = 0
        
        for file_path in all_merged_files:
            if file_path == demo_file:
                continue
                
            full_path = self.data_dir / file_path
            dataset_name = file_path.stem.replace('_merged', '')
            
            try:
                logger.info(f"📁 Loading {dataset_name}...")
                df_chunks = pd.read_csv(full_path, chunksize=10000)
                df = pd.concat(df_chunks, ignore_index=True)
                
                before_count = len(self.master_df)
                before_cols = self.master_df.shape[1]
                
                self.master_df = pd.merge(self.master_df, df, on=merge_keys, how='left')
                
                new_cols = self.master_df.shape[1] - before_cols
                logger.info(f"✅ {dataset_name} merged: +{new_cols} columns, {len(self.master_df):,} rows")
                successful_merges += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to load {dataset_name}: {e}")
                failed_merges += 1
                continue
        
        logger.info("="*60)
        logger.info(f"📊 Data merging summary:")
        logger.info(f"   ✅ Successfully merged: {successful_merges} datasets")
        logger.info(f"   ❌ Failed/skipped: {failed_merges} datasets")
        logger.info(f"   📏 Final dimensions: {self.master_df.shape[0]:,} rows × {self.master_df.shape[1]:,} columns")
        logger.info("="*60)
        
        return True
    
    def identify_analysis_variables(self):
        """识别并选择用于分析的变量"""
        logger.info("🔍 开始识别分析变量...")
        
        # 排除的列（ID、权重、元数据等）
        exclude_patterns = [
            'RespondentSequenceNumber', 'YearRange', 'Weight', 'PSU', 'Stratum',
            'Interview', 'Comment', 'Status', 'Flag', 'Code', 'ID', 'Cycle',
            'Sample', 'Variance', 'Exam', 'Component'
        ]
        
        # 筛选数值型变量
        numeric_columns = []
        for col in self.master_df.columns:
            # 跳过排除模式
            if any(pattern in col for pattern in exclude_patterns):
                continue
            
            # 尝试转换为数值型
            try:
                numeric_vals = pd.to_numeric(self.master_df[col], errors='coerce')
                non_null_count = numeric_vals.notna().sum()
                
                # 要求至少有1000个非空值
                if non_null_count >= 1000:
                    # 检查是否有足够的变异性
                    unique_count = numeric_vals.nunique()
                    if unique_count >= 3:  # 至少3个不同值
                        numeric_columns.append(col)
                        
            except:
                continue
        
        self.analysis_variables = numeric_columns
        
        # Categorize variables by type
        self.variable_categories = {
            'Demographics': [],
            'Dietary_Nutrition': [],
            'Lifestyle': [],
            'Disease_History': [],
            'Physical_Exam': [],
            'Laboratory': [],
            'Others': []
        }
        
        for var in self.analysis_variables:
            if any(keyword in var for keyword in ['Age', 'Gender', 'Race', 'Education', 'Income', 'Marital', 'Birth']):
                self.variable_categories['Demographics'].append(var)
            elif any(keyword in var for keyword in ['Energy', 'Protein', 'Fat', 'Fiber', 'Sugar', 'Sodium', 'Vitamin', 'Mineral', 'Dietary', 'Food']):
                self.variable_categories['Dietary_Nutrition'].append(var)
            elif any(keyword in var for keyword in ['Smoke', 'Alcohol', 'Physical', 'Activity', 'Exercise', 'Sleep']):
                self.variable_categories['Lifestyle'].append(var)
            elif any(keyword in var for keyword in ['diabetes', 'hypertension', 'Disease', 'told', 'Doctor', 'Medical']):
                self.variable_categories['Disease_History'].append(var)
            elif any(keyword in var for keyword in ['BMI', 'Weight', 'Height', 'Circumference', 'Blood Pressure', 'Body']):
                self.variable_categories['Physical_Exam'].append(var)
            elif any(keyword in var for keyword in ['Cholesterol', 'Glucose', 'Albumin', 'Creatinine', 'Hemoglobin', 'Laboratory', 'Urine', 'Blood']):
                self.variable_categories['Laboratory'].append(var)
            else:
                self.variable_categories['Others'].append(var)
        
        logger.info(f"📋 Variable categorization:")
        for category, vars_list in self.variable_categories.items():
            if vars_list:
                logger.info(f"   {category}: {len(vars_list)} variables")
        
        logger.info(f"🎯 Total identified {len(self.analysis_variables)} analysis variables")
        
        # 保存变量列表
        var_info = {
            'total_variables': len(self.analysis_variables),
            'categories': {k: len(v) for k, v in self.variable_categories.items()},
            'variable_list': self.analysis_variables,
            'detailed_categories': self.variable_categories
        }
        
        with open(self.output_dir / 'analysis_variables.json', 'w', encoding='utf-8') as f:
            json.dump(var_info, f, ensure_ascii=False, indent=2)
        
        return len(self.analysis_variables) > 0
    
    def clean_and_prepare_data(self):
        """Data cleaning and preprocessing"""
        logger.info("🧹 Starting data cleaning and preprocessing...")
        
        # Keep only analysis variables and merge keys
        keep_columns = self.analysis_variables + ['RespondentSequenceNumber', 'YearRange']
        self.master_df = self.master_df[keep_columns]
        
        logger.info(f"📊 Data dimensions after cleaning: {self.master_df.shape}")
        
        # Numeric conversion
        for var in self.analysis_variables:
            if var in self.master_df.columns:
                # Convert to numeric type
                self.master_df[var] = pd.to_numeric(self.master_df[var], errors='coerce')
                
                # Handle NHANES coded missing values
                missing_codes = [7, 9, 77, 99, 777, 999, 7777, 9999]
                for code in missing_codes:
                    self.master_df.loc[self.master_df[var] == code, var] = np.nan
        
        # Calculate missing value statistics for each variable
        missing_stats = {}
        for var in self.analysis_variables:
            total = len(self.master_df)
            missing = self.master_df[var].isna().sum()
            missing_rate = missing / total * 100
            missing_stats[var] = {
                'total': total,
                'missing': missing,
                'available': total - missing,
                'missing_rate': missing_rate
            }
        
        # Save missing value statistics
        missing_df = pd.DataFrame(missing_stats).T
        missing_df.to_csv(self.output_dir / 'missing_value_statistics.csv', encoding='utf-8-sig')
        
        logger.info(f"📈 Data quality statistics saved to: missing_value_statistics.csv")
        logger.info("✅ Data cleaning completed")
        
        return True
    
    def perform_regression_test(self, df, outcome_var, predictor_vars):
        """执行回归分析并返回详细结果"""
        try:
            # 准备数据
            y = df[outcome_var].dropna()
            X = df[predictor_vars].dropna()
            
            # 确保X和y的索引匹配
            common_idx = y.index.intersection(X.index)
            y = y.loc[common_idx]
            X = X.loc[common_idx]
            
            if len(y) < MIN_SAMPLE_SIZE:
                return None
            
            # 添加常数项
            X = sm.add_constant(X)
            
            # 检查结局变量类型
            unique_vals = y.nunique()
            if unique_vals == 2:
                # 二分类变量，使用逻辑回归
                model = sm.Logit(y, X)
            else:
                # 连续变量，使用线性回归
                model = sm.OLS(y, X)
            
            # 拟合模型
            try:
                result = model.fit(disp=0)
                
                # 返回第一个预测变量的结果
                if len(result.pvalues) > 1:
                    return {
                        'sample_size': len(y),
                        'p_value': result.pvalues.iloc[1],
                        'coefficient': result.params.iloc[1] if hasattr(result, 'params') else np.nan,
                        'std_error': result.bse.iloc[1] if hasattr(result, 'bse') else np.nan,
                        'r_squared': getattr(result, 'rsquared', np.nan),
                        'model_type': 'logistic' if unique_vals == 2 else 'linear'
                    }
                else:
                    return None
            except:
                return None
                
        except Exception as e:
            return None
    
    def analyze_chain_length(self, chain_length):
        """分析指定长度的因果链"""
        logger.info(f"🔗 Starting analysis of {chain_length}-variable causal chains...")
        
        chain_results = []
        total_combinations = 0
        processed_combinations = 0
        
        # 计算总组合数用于进度显示
        if chain_length <= len(self.analysis_variables):
            from math import factorial
            total_combinations = factorial(len(self.analysis_variables)) // factorial(len(self.analysis_variables) - chain_length)
            logger.info(f"📊 总共需要测试 {total_combinations:,} 个 {chain_length}-变量组合")
        
        # 生成所有可能的变量排列
        for var_combination in permutations(self.analysis_variables, chain_length):
            processed_combinations += 1
            
            if processed_combinations % 10000 == 0:
                progress = (processed_combinations / total_combinations * 100) if total_combinations > 0 else 0
                logger.info(f"📈 进度: {processed_combinations:,}/{total_combinations:,} ({progress:.1f}%)")
            
            # 创建分析数据集
            analysis_df = self.master_df[list(var_combination)].dropna()
            
            if len(analysis_df) < MIN_SAMPLE_SIZE:
                continue
            
            # 测试所有相邻变量对的关联
            chain_valid = True
            chain_stats = []
            
            for i in range(chain_length - 1):
                predictor = var_combination[i]
                outcome = var_combination[i + 1]
                
                # 执行回归分析
                reg_result = self.perform_regression_test(
                    analysis_df, outcome, [predictor]
                )
                
                if reg_result is None:
                    chain_valid = False
                    break
                
                chain_stats.append({
                    'step': i + 1,
                    'predictor': predictor,
                    'outcome': outcome,
                    'sample_size': reg_result['sample_size'],
                    'p_value': reg_result['p_value'],
                    'coefficient': reg_result['coefficient'],
                    'std_error': reg_result['std_error'],
                    'r_squared': reg_result['r_squared'],
                    'model_type': reg_result['model_type']
                })
            
            if chain_valid:
                # 计算整个链的综合统计
                all_p_values = [stat['p_value'] for stat in chain_stats]
                significant_links = sum(1 for p in all_p_values if p < ALPHA)
                
                chain_result = {
                    'chain_variables': list(var_combination),
                    'chain_length': chain_length,
                    'sample_size': min(stat['sample_size'] for stat in chain_stats),
                    'chain_stats': chain_stats,
                    'all_p_values': all_p_values,
                    'significant_links': significant_links,
                    'all_significant': significant_links == (chain_length - 1),
                    'geometric_mean_p': np.exp(np.mean(np.log(all_p_values))),
                    'min_p_value': min(all_p_values),
                    'max_p_value': max(all_p_values),
                    'chain_pathway': ' → '.join(var_combination)
                }
                
                chain_results.append(chain_result)
        
        logger.info(f"✅ {chain_length}-variable chain analysis completed: found {len(chain_results):,} valid chains")
        return chain_results
    
    def run_exhaustive_analysis(self):
        """Run exhaustive causal analysis"""
        logger.info("🚀 Starting exhaustive causal pathway analysis...")
        
        # Limit variable count to prevent computational explosion
        if len(self.analysis_variables) > 50:
            logger.warning(f"⚠️  Too many variables ({len(self.analysis_variables)}), randomly selecting 50 for analysis")
            import random
            random.seed(42)
            self.analysis_variables = random.sample(self.analysis_variables, 50)
        
        # Analyze different chain lengths
        for chain_length in range(3, MAX_CHAIN_LENGTH + 1):
            try:
                results = self.analyze_chain_length(chain_length)
                self.results[f'{chain_length}_chain'] = results
                
                # Save results immediately (prevent memory overflow)
                self.save_chain_results(chain_length, results)
                
                # Clear memory
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {chain_length}-variable chains: {e}")
                continue
        
        logger.info("🎉 Exhaustive analysis completed!")
        return True
    
    def save_chain_results(self, chain_length, results):
        """保存指定长度链的结果"""
        if not results:
            logger.info(f"⚠️  {chain_length}-变量链无结果可保存")
            return
        
        # 转换为平面DataFrame格式
        flat_results = []
        for result in results:
            base_info = {
                'chain_pathway': result['chain_pathway'],
                'chain_length': result['chain_length'],
                'sample_size': result['sample_size'],
                'significant_links': result['significant_links'],
                'all_significant': result['all_significant'],
                'geometric_mean_p': result['geometric_mean_p'],
                'min_p_value': result['min_p_value'],
                'max_p_value': result['max_p_value']
            }
            
            # 添加每一步的详细信息
            for i, step_stat in enumerate(result['chain_stats']):
                step_info = base_info.copy()
                step_info.update({
                    'step_number': step_stat['step'],
                    'step_predictor': step_stat['predictor'],
                    'step_outcome': step_stat['outcome'],
                    'step_p_value': step_stat['p_value'],
                    'step_coefficient': step_stat['coefficient'],
                    'step_std_error': step_stat['std_error'],
                    'step_r_squared': step_stat['r_squared'],
                    'step_model_type': step_stat['model_type']
                })
                flat_results.append(step_info)
        
        # 保存为CSV
        results_df = pd.DataFrame(flat_results)
        
        # 按显著性和p值排序
        results_df = results_df.sort_values([
            'all_significant', 'significant_links', 'geometric_mean_p'
        ], ascending=[False, False, True])
        
        output_file = self.output_dir / f'{chain_length}_variable_chains_complete_results.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"💾 {chain_length}-变量链结果已保存: {output_file}")
        logger.info(f"   📊 总计: {len(results):,} 个链，{len(flat_results):,} 个链步")
        
        # 保存显著性汇总
        summary = {
            'total_chains': len(results),
            'all_significant_chains': sum(1 for r in results if r['all_significant']),
            'partially_significant_chains': sum(1 for r in results if r['significant_links'] > 0 and not r['all_significant']),
            'non_significant_chains': sum(1 for r in results if r['significant_links'] == 0),
            'average_sample_size': np.mean([r['sample_size'] for r in results]),
            'min_sample_size': min(r['sample_size'] for r in results),
            'max_sample_size': max(r['sample_size'] for r in results)
        }
        
        summary_file = self.output_dir / f'{chain_length}_variable_chains_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 汇总统计:")
        logger.info(f"   🎯 完全显著链: {summary['all_significant_chains']:,}")
        logger.info(f"   ⚡ 部分显著链: {summary['partially_significant_chains']:,}")
        logger.info(f"   ❌ 非显著链: {summary['non_significant_chains']:,}")
    
    def generate_comprehensive_report(self):
        """生成全面的分析报告"""
        logger.info("📊 生成全面分析报告...")
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("NHANES 全变量穷举因果路径分析报告")
        report_lines.append("=" * 100)
        report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据集规模: {self.master_df.shape[0]:,} 行 × {len(self.analysis_variables)} 个分析变量")
        report_lines.append(f"最小样本量要求: {MIN_SAMPLE_SIZE:,}")
        report_lines.append(f"显著性水平: α = {ALPHA}")
        report_lines.append("")
        
        # 总结每种链长度的结果
        total_chains = 0
        total_significant = 0
        
        for chain_length in range(3, MAX_CHAIN_LENGTH + 1):
            results = self.results.get(f'{chain_length}_chain', [])
            if results:
                significant_count = sum(1 for r in results if r['all_significant'])
                total_chains += len(results)
                total_significant += significant_count
                
                report_lines.append(f"{chain_length}-变量因果链分析结果:")
                report_lines.append(f"  📊 总发现链数: {len(results):,}")
                report_lines.append(f"  🎯 完全显著链: {significant_count:,} ({significant_count/len(results)*100:.1f}%)")
                report_lines.append(f"  📈 部分显著链: {sum(1 for r in results if 0 < r['significant_links'] < chain_length-1):,}")
                report_lines.append(f"  ❌ 非显著链: {sum(1 for r in results if r['significant_links'] == 0):,}")
                report_lines.append("")
        
        report_lines.append("=" * 50)
        report_lines.append("总体统计汇总")
        report_lines.append("=" * 50)
        report_lines.append(f"🔢 总计发现因果链: {total_chains:,}")
        report_lines.append(f"⭐ 总计显著因果链: {total_significant:,}")
        report_lines.append(f"📊 显著率: {total_significant/total_chains*100:.2f}%" if total_chains > 0 else "📊 显著率: 0%")
        report_lines.append("")
        
        # 变量分类统计
        report_lines.append("📋 分析变量分类统计:")
        for category, vars_list in self.variable_categories.items():
            if vars_list:
                report_lines.append(f"  {category}: {len(vars_list)} 个变量")
        report_lines.append("")
        
        # 保存报告
        report_file = self.output_dir / 'comprehensive_analysis_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 同时打印到控制台
        print('\n'.join(report_lines))
        
        logger.info(f"📄 全面报告已保存: {report_file}")
    
    def create_summary_visualizations(self):
        """Create summary visualizations"""
        logger.info("📈 Creating summary visualizations...")
        
        # Create plot directory
        plot_dir = self.output_dir / 'summary_plots'
        plot_dir.mkdir(exist_ok=True)
        
        # 1. Chain length distribution plot
        chain_lengths = []
        total_counts = []
        significant_counts = []
        
        for chain_length in range(3, MAX_CHAIN_LENGTH + 1):
            results = self.results.get(f'{chain_length}_chain', [])
            if results:
                chain_lengths.append(f'{chain_length}-Variable Chain')
                total_counts.append(len(results))
                significant_counts.append(sum(1 for r in results if r['all_significant']))
        
        if chain_lengths:
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(chain_lengths))
            width = 0.35
            
            ax.bar(x - width/2, total_counts, width, label='Total Chains', alpha=0.8)
            ax.bar(x + width/2, significant_counts, width, label='Significant Chains', alpha=0.8)
            
            ax.set_xlabel('Causal Chain Length')
            ax.set_ylabel('Number of Chains')
            ax.set_title('Distribution of Causal Chains by Length')
            ax.set_xticks(x)
            ax.set_xticklabels(chain_lengths)
            ax.legend()
            
            # Add value labels
            for i, (total, sig) in enumerate(zip(total_counts, significant_counts)):
                ax.text(i - width/2, total + max(total_counts)*0.01, f'{total:,}', 
                       ha='center', va='bottom')
                ax.text(i + width/2, sig + max(total_counts)*0.01, f'{sig:,}', 
                       ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'chain_length_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Chain length distribution plot saved")
        
        # 2. Variable category participation analysis
        category_participation = {cat: 0 for cat in self.variable_categories.keys()}
        
        for chain_length in range(3, MAX_CHAIN_LENGTH + 1):
            results = self.results.get(f'{chain_length}_chain', [])
            for result in results:
                if result['all_significant']:  # Only count significant chains
                    for var in result['chain_variables']:
                        for cat, vars_list in self.variable_categories.items():
                            if var in vars_list:
                                category_participation[cat] += 1
                                break
        
        if any(category_participation.values()):
            fig, ax = plt.subplots(figsize=(12, 6))
            categories = list(category_participation.keys())
            counts = list(category_participation.values())
            
            bars = ax.bar(categories, counts)
            ax.set_xlabel('Variable Category')
            ax.set_ylabel('Occurrences in Significant Causal Chains')
            ax.set_title('Variable Category Participation in Significant Causal Chains')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                           f'{count:,}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plot_dir / 'category_participation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Variable category participation plot saved")
    
    def run_complete_analysis(self):
        """运行完整的穷举分析流程"""
        start_time = datetime.now()
        logger.info("🚀 Starting NHANES exhaustive causal pathway analysis...")
        logger.info(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Load and merge data
            if not self.load_and_merge_data():
                logger.error("❌ Data loading failed, analysis terminated")
                return False
            
            # Step 2: Identify analysis variables
            if not self.identify_analysis_variables():
                logger.error("❌ Variable identification failed, analysis terminated")
                return False
            
            # Step 3: Data cleaning and preprocessing
            if not self.clean_and_prepare_data():
                logger.error("❌ Data preprocessing failed, analysis terminated")
                return False
            
            # Step 4: Run exhaustive analysis
            if not self.run_exhaustive_analysis():
                logger.error("❌ Exhaustive analysis failed, analysis terminated")
                return False
            
            # Step 5: Generate reports and visualizations
            self.generate_comprehensive_report()
            self.create_summary_visualizations()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("🎉 Exhaustive variable analysis completed!")
            logger.info(f"⏱️  Total duration: {duration}")
            logger.info(f"📁 All results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error during analysis: {e}")
            return False

def main():
    """Main function"""
    print("🔬 NHANES Exhaustive Causal Pathway Analysis")
    print("=" * 80)
    print("⚠️  WARNING: This analysis tests ALL possible variable combinations - extremely computationally intensive!")
    print("💡 Recommended to run in high-performance computing environment")
    print("=" * 80)
    
    # Check data directory
    if not Path(DATA_DIR).exists():
        print(f"❌ Error: Data directory does not exist: {DATA_DIR}")
        return
    
    # Create analysis instance and run
    analyzer = NHANESExhaustiveAnalysis()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✅ Exhaustive analysis completed successfully!")
        print(f"📁 Results saved to: {analyzer.output_dir}")
        print("📊 Please review the detailed CSV files, JSON statistics, and plots")
        print("🔍 Recommend using Excel or data analysis tools for further exploration")
    else:
        print("\n❌ Errors occurred during analysis, please check log files")

if __name__ == "__main__":
    main() 