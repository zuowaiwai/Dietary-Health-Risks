#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHANES数据预处理脚本
===================

功能描述:
- 递归扫描NHANES数据目录，查找所有.XPT文件
- 将.XPT文件转换为CSV格式
- 标准化列名以提高可读性和一致性
- 按类别合并不同年份的数据文件
- 生成便于分析的结构化CSV文件

作者: 自动生成
版本: 1.0
日期: 2024
"""

import os
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
from typing import List, Dict, Optional, Tuple

# 忽略SAS文件读取警告
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== 配置区域 ====================

# 输入和输出目录配置
INPUT_ROOT_DIR = "NHANES_Data"
OUTPUT_ROOT_DIR = "NHANES_Processed"

# 列名标准化映射字典
COLUMN_RENAME_MAP = {
    # 核心标识符
    'SEQN': 'ID',                           # 参与者唯一标识符
    'SDDSRVYR': 'DataReleaseCycle',         # 数据发布周期
    
    # 人口统计学变量
    'RIAGENDR': 'Gender',                   # 性别
    'RIDAGEYR': 'AgeInYears',              # 年龄（年）
    'RIDAGEMN': 'AgeInMonths',             # 年龄（月）
    'RIDRETH3': 'RaceEthnicity',           # 种族/族裔
    'RIDRETH1': 'RaceEthnicity_Old',       # 种族/族裔（旧版）
    'DMDBORN4': 'CountryOfBirth',          # 出生国家
    'DMDCITZN': 'CitizenshipStatus',       # 公民身份
    'DMDEDUC3': 'Education3to5',           # 教育水平（3-5岁）
    'DMDEDUC2': 'EducationAdults',         # 教育水平（成人）
    'DMDMARTL': 'MaritalStatus',           # 婚姻状况
    'DMDFMSIZ': 'FamilySize',              # 家庭人数
    'INDHHIN2': 'HouseholdIncome',         # 家庭收入
    'INDFMIN2': 'FamilyIncome',            # 家庭收入详细
    'INDFMPIR': 'PovertyIncomeRatio',      # 贫困收入比
    
    # 身体测量变量
    'BMXWT': 'Weight_kg',                  # 体重（公斤）
    'BMXRECUM': 'RecumbentLength_cm',      # 卧位身长（厘米）
    'BMXHEAD': 'HeadCircumference_cm',     # 头围（厘米）
    'BMXHT': 'Height_cm',                  # 身高（厘米）
    'BMXBMI': 'BMI',                       # 身体质量指数
    'BMXLEG': 'UpperLegLength_cm',         # 大腿长度（厘米）
    'BMXARML': 'ArmLength_cm',             # 臂长（厘米）
    'BMXARMC': 'ArmCircumference_cm',      # 臂围（厘米）
    'BMXWAIST': 'WaistCircumference_cm',   # 腰围（厘米）
    'BMXHIP': 'HipCircumference_cm',       # 臀围（厘米）
    
    # 血压变量
    'BPXSY1': 'SystolicBP_1st_mmHg',      # 第1次收缩压
    'BPXDI1': 'DiastolicBP_1st_mmHg',     # 第1次舒张压
    'BPXSY2': 'SystolicBP_2nd_mmHg',      # 第2次收缩压
    'BPXDI2': 'DiastolicBP_2nd_mmHg',     # 第2次舒张压
    'BPXSY3': 'SystolicBP_3rd_mmHg',      # 第3次收缩压
    'BPXDI3': 'DiastolicBP_3rd_mmHg',     # 第3次舒张压
    'BPXSY4': 'SystolicBP_4th_mmHg',      # 第4次收缩压
    'BPXDI4': 'DiastolicBP_4th_mmHg',     # 第4次舒张压
    
    # 实验室检测变量
    'LBXGLU': 'Glucose_mg_dL',             # 血糖
    'LBXIN': 'Insulin_uU_mL',              # 胰岛素
    'LBXTC': 'TotalCholesterol_mg_dL',     # 总胆固醇
    'LBXHDL': 'HDL_Cholesterol_mg_dL',     # 高密度脂蛋白胆固醇
    'LBXLDL': 'LDL_Cholesterol_mg_dL',     # 低密度脂蛋白胆固醇
    'LBXTR': 'Triglycerides_mg_dL',        # 甘油三酯
    'LBXHGB': 'Hemoglobin_g_dL',          # 血红蛋白
    'LBXHCT': 'Hematocrit_percent',        # 血细胞比容
    'LBXWBCSI': 'WhiteBloodCells_SI',      # 白细胞计数
    'LBXRBCSI': 'RedBloodCells_SI',        # 红细胞计数
    
    # 营养素摄入变量
    'DR1TKCAL': 'Energy_Day1_kcal',        # 第一天能量摄入
    'DR1TPROT': 'Protein_Day1_g',          # 第一天蛋白质摄入
    'DR1TCARB': 'Carbohydrate_Day1_g',     # 第一天碳水化合物摄入
    'DR1TFAT': 'Fat_Day1_g',               # 第一天脂肪摄入
    'DR1TSFAT': 'SaturatedFat_Day1_g',     # 第一天饱和脂肪摄入
    'DR1TMFAT': 'MonounsaturatedFat_Day1_g', # 第一天单不饱和脂肪
    'DR1TPFAT': 'PolyunsaturatedFat_Day1_g', # 第一天多不饱和脂肪
    'DR1TCHOL': 'DietaryCholesterol_Day1_mg', # 第一天膳食胆固醇
    'DR1TLYCO': 'Lycopene_Day1_mcg',       # 第一天番茄红素
    'DR1TLZ': 'Lutein_Zeaxanthin_Day1_mcg', # 第一天叶黄素+玉米黄质
    'DR1TVB1': 'ThiaminB1_Day1_mg',        # 第一天维生素B1
    'DR1TVB2': 'RiboflavinB2_Day1_mg',     # 第一天维生素B2
    'DR1TNIAC': 'Niacin_Day1_mg',          # 第一天烟酸
    'DR1TVB6': 'VitaminB6_Day1_mg',        # 第一天维生素B6
    'DR1TFOLA': 'Folate_Day1_mcg',         # 第一天叶酸
    'DR1TVB12': 'VitaminB12_Day1_mcg',     # 第一天维生素B12
    'DR1TVC': 'VitaminC_Day1_mg',          # 第一天维生素C
    'DR1TVK': 'VitaminK_Day1_mcg',         # 第一天维生素K
    'DR1TCALC': 'Calcium_Day1_mg',         # 第一天钙摄入
    'DR1TPHOS': 'Phosphorus_Day1_mg',      # 第一天磷摄入
    'DR1TMAGN': 'Magnesium_Day1_mg',       # 第一天镁摄入
    'DR1TIRON': 'Iron_Day1_mg',            # 第一天铁摄入
    'DR1TZINC': 'Zinc_Day1_mg',            # 第一天锌摄入
    'DR1TCOPP': 'Copper_Day1_mg',          # 第一天铜摄入
    'DR1TSODI': 'Sodium_Day1_mg',          # 第一天钠摄入
    'DR1TPOTA': 'Potassium_Day1_mg',       # 第一天钾摄入
    'DR1TSELE': 'Selenium_Day1_mcg',       # 第一天硒摄入
    'DR1TCAFF': 'Caffeine_Day1_mg',        # 第一天咖啡因摄入
    'DR1TTHEO': 'Theobromine_Day1_mg',     # 第一天可可碱摄入
    'DR1TALCO': 'Alcohol_Day1_g',          # 第一天酒精摄入
    
    # 样本权重变量
    'WTINT2YR': 'InterviewWeight_2Year',   # 2年访谈权重
    'WTMEC2YR': 'MobileExamWeight_2Year',  # 2年体检权重
    'WTMEC4YR': 'MobileExamWeight_4Year',  # 4年体检权重
    'WTSAF2YR': 'FastingSubsampleWeight_2Year', # 2年空腹亚样本权重
    
    # 其他常用变量
    'RIDEXPRG': 'PregnancyStatus',         # 怀孕状况
    'DMQMILIT': 'MilitaryService',         # 军队服役状况
    'DMDHHSIZ': 'HouseholdSize',           # 家庭人数
    'DMDHHSZA': 'HouseholdSizeCategory_A', # 家庭人数类别A
    'DMDHHSZB': 'HouseholdSizeCategory_B', # 家庭人数类别B
    'DMDHRGND': 'HouseholdReferencePersonGender', # 家庭参考人性别
    'DMDHRAGE': 'HouseholdReferencePersonAge',    # 家庭参考人年龄
    'DMDHRBR4': 'HouseholdReferencePersonCountryOfBirth', # 家庭参考人出生国
    'DMDHREDU': 'HouseholdReferencePersonEducation',      # 家庭参考人教育
    'DMDHRMAR': 'HouseholdReferencePersonMaritalStatus',  # 家庭参考人婚姻状况
}

# ==================== 日志配置 ====================

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nhanes_preprocessing.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==================== 工具函数 ====================

def extract_survey_cycle(file_path: str) -> Optional[str]:
    """
    从文件路径中提取调查周期信息
    
    Args:
        file_path: 文件完整路径
        
    Returns:
        调查周期字符串，如 "2013-2014"，如果提取失败返回None
    """
    # 尝试多种模式匹配调查周期
    patterns = [
        r'(\d{4}-\d{4})',           # 2013-2014格式
        r'(\d{8})',                 # 20132014格式
        r'_([HI])\.XPT',            # 文件名中的周期代码
        r'_(\d{4}_\d{4})',          # 2013_2014格式
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, file_path, re.IGNORECASE)
        if matches:
            cycle = matches.group(1)
            
            # 处理不同格式
            if len(cycle) == 8 and cycle.isdigit():  # 20132014格式
                year1 = cycle[:4]
                year2 = cycle[4:]
                return f"{year1}-{year2}"
            elif cycle in ['H', 'I', 'J', 'K', 'L', 'M']:  # 字母代码
                cycle_map = {
                    'H': '2013-2014',
                    'I': '2015-2016', 
                    'J': '2017-2018',
                    'K': '2019-2020',
                    'L': '2021-2022',
                    'M': '2023-2024'
                }
                return cycle_map.get(cycle)
            elif '-' in cycle:  # 已经是正确格式
                return cycle
            elif '_' in cycle:  # 下划线格式
                return cycle.replace('_', '-')
    
    # 如果无法从文件名提取，尝试从路径提取
    path_parts = file_path.split(os.sep)
    for part in path_parts:
        if re.match(r'\d{4}-\d{4}', part):
            return part
        elif re.match(r'\d{8}', part):
            year1 = part[:4]
            year2 = part[4:]
            return f"{year1}-{year2}"
    
    return None

def safe_read_xpt(file_path: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    安全地读取XPT文件
    
    Args:
        file_path: XPT文件路径
        logger: 日志记录器
        
    Returns:
        pandas DataFrame或None（如果读取失败）
    """
    try:
        # 尝试读取XPT文件
        df = pd.read_sas(file_path, format='xport', encoding='utf-8')
        
        if df.empty:
            logger.warning(f"文件为空: {file_path}")
            return None
            
        logger.debug(f"成功读取文件: {file_path} (形状: {df.shape})")
        return df
        
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {str(e)}")
        return None

def standardize_column_names(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    标准化列名
    
    Args:
        df: 原始DataFrame
        logger: 日志记录器
        
    Returns:
        列名标准化后的DataFrame
    """
    df_copy = df.copy()
    
    # 应用列名映射
    renamed_columns = {}
    for old_name, new_name in COLUMN_RENAME_MAP.items():
        if old_name in df_copy.columns:
            renamed_columns[old_name] = new_name
    
    if renamed_columns:
        df_copy.rename(columns=renamed_columns, inplace=True)
        logger.debug(f"重命名了 {len(renamed_columns)} 个列: {list(renamed_columns.keys())}")
    
    # 将所有列名转换为字符串并清理
    df_copy.columns = [str(col).strip() for col in df_copy.columns]
    
    return df_copy

def create_output_directory(output_path: str) -> None:
    """
    创建输出目录
    
    Args:
        output_path: 输出路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

def find_xpt_files_by_category(input_root: str, logger: logging.Logger) -> Dict[str, List[Tuple[str, str]]]:
    """
    按类别查找XPT文件
    
    Args:
        input_root: 输入根目录
        logger: 日志记录器
        
    Returns:
        字典，键为类别路径，值为(文件路径, 调查周期)的列表
    """
    categories = {}
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.upper().endswith('.XPT'):
                file_path = os.path.join(root, file)
                
                # 提取调查周期
                survey_cycle = extract_survey_cycle(file_path)
                if not survey_cycle:
                    logger.warning(f"无法提取调查周期: {file_path}")
                    continue
                
                # 确定类别
                # 从根目录到文件所在目录的相对路径
                rel_path = os.path.relpath(root, input_root)
                path_parts = rel_path.split(os.sep)
                
                # 找到最深层的有意义的类别目录
                category_path = None
                for i in range(len(path_parts)):
                    potential_category = os.sep.join(path_parts[:i+1])
                    # 检查这个路径是否包含多个子目录（表明它是一个类别）
                    full_potential_path = os.path.join(input_root, potential_category)
                    if os.path.exists(full_potential_path):
                        subdirs = [d for d in os.listdir(full_potential_path) 
                                 if os.path.isdir(os.path.join(full_potential_path, d))]
                        # 如果有年份目录或其他子目录，这可能是个类别
                        if any(re.match(r'\d{4}-\d{4}', d) for d in subdirs) or len(subdirs) > 1:
                            category_path = potential_category
                
                if not category_path:
                    # 如果找不到明确的类别，使用文件所在的直接父目录
                    category_path = path_parts[-1] if path_parts and path_parts[0] != '.' else "未分类"
                
                # 将文件添加到对应类别
                if category_path not in categories:
                    categories[category_path] = []
                
                categories[category_path].append((file_path, survey_cycle))
                logger.debug(f"文件归类: {file} -> {category_path} ({survey_cycle})")
    
    return categories

def process_category(category_name: str, file_list: List[Tuple[str, str]], 
                    output_root: str, logger: logging.Logger) -> bool:
    """
    处理单个类别的所有文件
    
    Args:
        category_name: 类别名称
        file_list: (文件路径, 调查周期) 的列表
        output_root: 输出根目录
        logger: 日志记录器
        
    Returns:
        处理是否成功
    """
    if not file_list:
        logger.warning(f"类别 '{category_name}' 没有找到任何文件")
        return False
    
    logger.info(f"开始处理类别: {category_name} (共 {len(file_list)} 个文件)")
    
    dataframes = []
    
    # 处理每个文件
    for file_path, survey_cycle in tqdm(file_list, desc=f"处理 {category_name}", leave=False):
        # 读取XPT文件
        df = safe_read_xpt(file_path, logger)
        if df is None:
            continue
        
        # 添加调查周期列
        df['Years'] = survey_cycle
        
        # 标准化列名
        df = standardize_column_names(df, logger)
        
        # 添加到列表
        dataframes.append(df)
        logger.debug(f"已处理: {os.path.basename(file_path)} ({survey_cycle}) - 形状: {df.shape}")
    
    if not dataframes:
        logger.error(f"类别 '{category_name}' 没有成功处理任何文件")
        return False
    
    # 合并所有DataFrame
    try:
        logger.info(f"正在合并 {len(dataframes)} 个DataFrame...")
        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
        logger.info(f"合并完成，最终形状: {merged_df.shape}")
        
        # 重新排列列，将Years和ID列放在前面
        columns = list(merged_df.columns)
        priority_columns = ['Years', 'ID']
        
        # 构建新的列顺序
        new_column_order = []
        for col in priority_columns:
            if col in columns:
                new_column_order.append(col)
                columns.remove(col)
        new_column_order.extend(sorted(columns))
        
        merged_df = merged_df[new_column_order]
        
    except Exception as e:
        logger.error(f"合并DataFrame失败: {str(e)}")
        return False
    
    # 构建输出文件路径
    safe_category_name = category_name.replace(os.sep, '_').replace('/', '_')
    output_filename = f"{safe_category_name}.csv"
    output_path = os.path.join(output_root, output_filename)
    
    # 创建输出目录
    create_output_directory(output_path)
    
    # 保存到CSV
    try:
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"成功保存: {output_path} (形状: {merged_df.shape})")
        
        # 打印一些统计信息
        logger.info(f"  - 包含的调查周期: {sorted(merged_df['Years'].unique())}")
        logger.info(f"  - 总记录数: {len(merged_df)}")
        logger.info(f"  - 总列数: {len(merged_df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"保存文件失败 {output_path}: {str(e)}")
        return False

def print_processing_summary(categories: Dict[str, List[Tuple[str, str]]], logger: logging.Logger):
    """
    打印处理摘要
    
    Args:
        categories: 按类别分组的文件字典
        logger: 日志记录器
    """
    logger.info("\n" + "="*80)
    logger.info("NHANES数据预处理摘要")
    logger.info("="*80)
    
    total_files = sum(len(files) for files in categories.values())
    logger.info(f"发现的类别总数: {len(categories)}")
    logger.info(f"待处理文件总数: {total_files}")
    
    logger.info("\n按类别分组的文件统计:")
    for category, files in sorted(categories.items()):
        cycles = set(cycle for _, cycle in files)
        logger.info(f"  {category}: {len(files)} 个文件, 覆盖 {len(cycles)} 个调查周期")
        logger.info(f"    调查周期: {sorted(cycles)}")
    
    logger.info("="*80)

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始NHANES数据预处理...")
    logger.info(f"输入目录: {INPUT_ROOT_DIR}")
    logger.info(f"输出目录: {OUTPUT_ROOT_DIR}")
    
    # 检查输入目录
    if not os.path.exists(INPUT_ROOT_DIR):
        logger.error(f"输入目录不存在: {INPUT_ROOT_DIR}")
        return
    
    # 创建输出目录
    Path(OUTPUT_ROOT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 按类别查找所有XPT文件
    logger.info("正在扫描目录结构...")
    categories = find_xpt_files_by_category(INPUT_ROOT_DIR, logger)
    
    if not categories:
        logger.error("未找到任何XPT文件")
        return
    
    # 打印处理摘要
    print_processing_summary(categories, logger)
    
    # 处理每个类别
    successful_categories = 0
    failed_categories = 0
    
    logger.info("\n开始处理数据...")
    
    for category_name in tqdm(sorted(categories.keys()), desc="处理类别"):
        file_list = categories[category_name]
        
        try:
            success = process_category(category_name, file_list, OUTPUT_ROOT_DIR, logger)
            if success:
                successful_categories += 1
            else:
                failed_categories += 1
        except Exception as e:
            logger.error(f"处理类别 '{category_name}' 时发生错误: {str(e)}")
            failed_categories += 1
    
    # 打印最终结果
    logger.info("\n" + "="*80)
    logger.info("数据预处理完成！")
    logger.info("="*80)
    logger.info(f"成功处理的类别: {successful_categories}")
    logger.info(f"失败的类别: {failed_categories}")
    logger.info(f"总类别数: {len(categories)}")
    logger.info(f"输出目录: {os.path.abspath(OUTPUT_ROOT_DIR)}")
    
    if successful_categories > 0:
        logger.info("\n生成的CSV文件:")
        for filename in sorted(os.listdir(OUTPUT_ROOT_DIR)):
            if filename.endswith('.csv'):
                file_path = os.path.join(OUTPUT_ROOT_DIR, filename)
                try:
                    # 快速读取文件信息
                    temp_df = pd.read_csv(file_path, nrows=0)  # 只读取列名
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    logger.info(f"  {filename}: {len(temp_df.columns)} 列, {file_size:.2f} MB")
                except Exception as e:
                    logger.warning(f"  {filename}: 无法读取文件信息 - {str(e)}")
    
    logger.info("="*80)

if __name__ == "__main__":
    main() 