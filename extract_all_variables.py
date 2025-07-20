#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取所有NHANES合并CSV文件的变量名
========================================

此脚本读取指定目录下所有CSV文件的列名，并汇总到一个CSV文件中。
避免加载完整数据，只读取列名以节省内存。

作者: AI Assistant
日期: 2024年
"""

import pandas as pd
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('variable_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_all_variables(input_dir, output_file):
    """
    提取所有CSV文件的变量名
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出CSV文件路径
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return False
    
    # 存储所有变量信息
    all_variables = []
    
    # 获取所有CSV文件
    csv_files = list(input_path.glob("*.csv"))
    logger.info(f"发现 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        try:
            logger.info(f"处理文件: {csv_file.name}")
            
            # 只读取第一行来获取列名，避免内存问题
            df_header = pd.read_csv(csv_file, nrows=0)
            columns = df_header.columns.tolist()
            
            # 获取文件大小
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            
            # 为每个变量创建记录
            for col_index, col_name in enumerate(columns):
                variable_info = {
                    'file_name': csv_file.name,
                    'file_size_mb': round(file_size_mb, 2),
                    'variable_index': col_index,
                    'variable_name': col_name,
                    'dataset_category': csv_file.stem.replace('_merged', ''),
                }
                all_variables.append(variable_info)
            
            logger.info(f"  ✅ 提取了 {len(columns)} 个变量")
            
        except Exception as e:
            logger.error(f"  ❌ 处理文件 {csv_file.name} 时出错: {e}")
            continue
    
    # 创建汇总DataFrame
    if all_variables:
        variables_df = pd.DataFrame(all_variables)
        
        # 按文件名和变量索引排序
        variables_df = variables_df.sort_values(['file_name', 'variable_index'])
        
        # 添加全局变量编号
        variables_df['global_variable_id'] = range(1, len(variables_df) + 1)
        
        # 重新排列列顺序
        column_order = [
            'global_variable_id',
            'file_name', 
            'dataset_category',
            'file_size_mb',
            'variable_index',
            'variable_name'
        ]
        variables_df = variables_df[column_order]
        
        # 保存到CSV
        variables_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 生成统计摘要
        summary_stats = {
            'total_files': len(csv_files),
            'total_variables': len(variables_df),
            'files_processed': variables_df['file_name'].nunique(),
            'largest_file': variables_df.loc[variables_df['file_size_mb'].idxmax()],
            'most_variables_file': variables_df.groupby('file_name').size().idxmax(),
            'variables_by_file': variables_df.groupby('file_name').size().to_dict()
        }
        
        logger.info("="*60)
        logger.info("📊 变量提取完成统计:")
        logger.info(f"   📁 处理文件数: {summary_stats['files_processed']}/{summary_stats['total_files']}")
        logger.info(f"   🔢 总变量数: {summary_stats['total_variables']:,}")
        logger.info(f"   📈 最大文件: {summary_stats['largest_file']['file_name']} ({summary_stats['largest_file']['file_size_mb']:.1f} MB)")
        logger.info(f"   📋 变量最多文件: {summary_stats['most_variables_file']} ({summary_stats['variables_by_file'][summary_stats['most_variables_file']]} 个变量)")
        logger.info(f"   💾 结果保存至: {output_file}")
        logger.info("="*60)
        
        # 保存统计摘要
        summary_file = str(output_file).replace('.csv', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("NHANES变量提取统计摘要\n")
            f.write("="*50 + "\n")
            f.write(f"处理时间: {pd.Timestamp.now()}\n")
            f.write(f"输入目录: {input_dir}\n")
            f.write(f"输出文件: {output_file}\n\n")
            f.write(f"文件处理情况:\n")
            f.write(f"  - 发现文件数: {summary_stats['total_files']}\n")
            f.write(f"  - 成功处理: {summary_stats['files_processed']}\n")
            f.write(f"  - 失败/跳过: {summary_stats['total_files'] - summary_stats['files_processed']}\n\n")
            f.write(f"变量统计:\n")
            f.write(f"  - 总变量数: {summary_stats['total_variables']:,}\n")
            f.write(f"  - 平均每文件变量数: {summary_stats['total_variables']/summary_stats['files_processed']:.1f}\n\n")
            f.write(f"各文件变量数统计:\n")
            for file_name, var_count in sorted(summary_stats['variables_by_file'].items()):
                f.write(f"  - {file_name}: {var_count} 个变量\n")
        
        return True
    else:
        logger.error("❌ 未能提取到任何变量")
        return False

def main():
    """主函数"""
    print("🔍 NHANES变量名提取工具")
    print("="*50)
    
    # 设置路径
    input_directory = "./Nhanes_processed_csv_merged"
    output_csv = "all_nhanes_variables.csv"
    
    # 检查输入目录
    if not Path(input_directory).exists():
        print(f"❌ 错误: 输入目录不存在: {input_directory}")
        print("请确保目录路径正确")
        return
    
    print(f"📁 输入目录: {input_directory}")
    print(f"📄 输出文件: {output_csv}")
    print("-"*50)
    
    # 执行提取
    success = extract_all_variables(input_directory, output_csv)
    
    if success:
        print("\n✅ 变量提取成功完成!")
        print(f"📊 所有变量信息已保存至: {output_csv}")
        print("💡 提示: 可以用Excel或其他工具打开查看结果")
    else:
        print("\n❌ 变量提取过程中出现错误")
        print("📝 请检查日志文件 variable_extraction.log 获取详细信息")

if __name__ == "__main__":
    main() 