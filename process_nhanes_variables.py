import os
import json
import pandas as pd
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def determine_variable_type(values_csv_path):
    """
    根据values CSV文件判断变量类型
    如果Value Description中有"Range of Values"则为Quantitative，否则为Categorical
    """
    try:
        if not os.path.exists(values_csv_path):
            return "Unknown"
        
        df = pd.read_csv(values_csv_path)
        
        # 检查Value Description列是否存在"Range of Values"
        if 'Value Description' in df.columns:
            value_descriptions = df['Value Description'].astype(str).str.strip()
            if any('Range of Values' in desc for desc in value_descriptions):
                return "Quantitative"
            else:
                return "Categorical"
        
        return "Unknown"
    except Exception as e:
        logger.warning(f"读取 {values_csv_path} 时出错: {e}")
        return "Unknown"

def process_nhanes_variables():
    """
    处理NHANES变量目录，提取变量信息并合并相同变量名的记录
    """
    variables_dir = Path("NHANES_Variables")
    variable_records = {}  # 用字典存储变量记录，key为变量名
    
    if not variables_dir.exists():
        logger.error("NHANES_Variables目录不存在")
        return
    
    total_datasets = 0
    total_variables = 0
    processed_variables = 0
    
    # 遍历所有数据集目录（如ALQ_H, DEMO_J等）
    for dataset_dir in variables_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
            
        total_datasets += 1
        logger.info(f"处理数据集: {dataset_dir.name}")
        
        # 遍历数据集下的所有变量目录
        for variable_dir in dataset_dir.iterdir():
            if not variable_dir.is_dir() or variable_dir.name.startswith('.'):
                continue
                
            total_variables += 1
            variable_name = variable_dir.name
            
            # 构建文件路径
            info_json_path = variable_dir / f"{variable_name}_info.json"
            values_csv_path = variable_dir / f"{variable_name}_values.csv"
            
            # 读取变量信息
            try:
                if info_json_path.exists():
                    with open(info_json_path, 'r', encoding='utf-8') as f:
                        variable_info = json.load(f)
                    
                    var_name = variable_info.get('Variable_Name', variable_name)
                    english_text = variable_info.get('English_Text', '')
                    label = variable_info.get('Label', '')
                    
                    # 判断变量类型
                    variable_type = determine_variable_type(values_csv_path)
                    
                    # 检查是否已经存在相同变量名的记录
                    if var_name in variable_records:
                        # 合并数据集信息
                        existing_datasets = variable_records[var_name]['Datasets']
                        if dataset_dir.name not in existing_datasets:
                            existing_datasets.append(dataset_dir.name)
                        
                        # 如果当前记录的信息更完整，则更新
                        if (not variable_records[var_name]['English_Text'] and english_text) or \
                           (variable_records[var_name]['Type'] == 'Unknown' and variable_type != 'Unknown'):
                            variable_records[var_name].update({
                                'English_Text': english_text or variable_records[var_name]['English_Text'],
                                'Label': label or variable_records[var_name]['Label'],
                                'Type': variable_type if variable_type != 'Unknown' else variable_records[var_name]['Type']
                            })
                    else:
                        # 新变量，直接添加
                        variable_records[var_name] = {
                            'var_name': var_name,
                            'English_Text': english_text,
                            'Label': label,
                            'Type': variable_type,
                            'Datasets': [dataset_dir.name]
                        }
                    
                    processed_variables += 1
                    
                    if processed_variables % 100 == 0:
                        logger.info(f"已处理 {processed_variables} 个变量...")
                        
                else:
                    logger.warning(f"未找到信息文件: {info_json_path}")
                    
            except Exception as e:
                logger.error(f"处理变量 {variable_name} 时出错: {e}")
                continue
    
    logger.info(f"处理完成: {total_datasets} 个数据集, {total_variables} 个变量, 成功处理 {processed_variables} 个")
    
    # 转换为结果列表并创建DataFrame
    if variable_records:
        results = []
        for var_name, record in variable_records.items():
            # 将数据集列表转换为字符串
            datasets_str = '; '.join(sorted(record['Datasets']))
            results.append({
                'var_name': record['var_name'],
                'English_Text': record['English_Text'],
                'Label': record['Label'],
                'Type': record['Type'],
                'Datasets': datasets_str,
                'Dataset_Count': len(record['Datasets'])
            })
        
        df = pd.DataFrame(results)
        
        # 按变量名排序
        df = df.sort_values('var_name').reset_index(drop=True)
        
        # 保存详细结果
        output_file = "nhanes_variables_merged.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"合并结果已保存到: {output_file}")
        
        # 创建只包含请求列的简化版本
        simplified_df = df[['var_name', 'English_Text', 'Type']].copy()
        simplified_output = "nhanes_variables_merged_summary.csv"
        simplified_df.to_csv(simplified_output, index=False, encoding='utf-8')
        logger.info(f"简化结果已保存到: {simplified_output}")
        
        # 统计信息
        unique_variables = len(df)
        total_original_records = sum(df['Dataset_Count'])
        type_counts = df['Type'].value_counts()
        
        logger.info(f"合并后统计信息:")
        logger.info(f"  唯一变量数: {unique_variables}")
        logger.info(f"  原始记录数: {total_original_records}")
        logger.info(f"  压缩比: {total_original_records/unique_variables:.1f}:1")
        logger.info("变量类型统计:")
        for var_type, count in type_counts.items():
            logger.info(f"  {var_type}: {count}")
        
        # 显示出现在多个数据集中的变量
        multi_dataset_vars = df[df['Dataset_Count'] > 1].sort_values('Dataset_Count', ascending=False)
        if len(multi_dataset_vars) > 0:
            logger.info(f"出现在多个数据集中的变量数: {len(multi_dataset_vars)}")
            logger.info("前10个出现最多的变量:")
            for _, row in multi_dataset_vars.head(10).iterrows():
                logger.info(f"  {row['var_name']}: {row['Dataset_Count']}个数据集 ({row['Datasets'][:100]}...)")
        
        return df
    else:
        logger.error("没有找到任何变量数据")
        return None

if __name__ == "__main__":
    logger.info("开始处理NHANES变量数据（合并相同变量名）...")
    result_df = process_nhanes_variables()
    
    if result_df is not None:
        logger.info("处理完成!")
        print(f"\n前10条记录预览:")
        print(result_df[['var_name', 'Type', 'Dataset_Count']].head(10))
    else:
        logger.error("处理失败!") 