#!/usr/bin/env python3
"""
validate_data_format.py

快速数据格式验证脚本
用于在运行主分析前检查数据格式是否正确
"""

import pandas as pd
import os
from pathlib import Path
import argparse
import sys

def validate_variable_dictionary(dict_path: str) -> bool:
    """验证变量字典格式"""
    print(f"🔍 检查变量字典: {dict_path}")
    
    if not os.path.exists(dict_path):
        print(f"❌ 错误: 变量字典文件不存在: {dict_path}")
        return False
    
    try:
        df = pd.read_csv(dict_path)
        print(f"✅ 成功加载变量字典，共 {len(df)} 行")
        
        # 检查必需的列
        required_cols = ['global_variable_id', 'file_name', 'variable_name', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ 错误: 缺少必需的列: {missing_cols}")
            print(f"   实际列名: {list(df.columns)}")
            return False
        
        print(f"✅ 包含所有必需的列: {required_cols}")
        
        # 检查B和B&C类型变量
        b_vars = df[df['category'].isin(['B', 'B&C'])]
        print(f"✅ 找到 {len(b_vars)} 个B/B&C类型变量")
        
        if len(b_vars) == 0:
            print("⚠️  警告: 没有找到B或B&C类型的变量")
            return False
        
        # 显示分类统计
        category_counts = df['category'].value_counts()
        print("📊 变量类型分布:")
        for cat, count in category_counts.head(10).items():
            print(f"   {cat}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: 无法读取变量字典: {e}")
        return False

def validate_data_directory(data_dir: str, variable_dict: pd.DataFrame = None) -> bool:
    """验证数据目录格式"""
    print(f"\n🔍 检查数据目录: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 错误: 数据目录不存在: {data_dir}")
        return False
    
    # 获取所有CSV文件
    csv_files = list(data_path.glob("*.csv"))
    print(f"✅ 找到 {len(csv_files)} 个CSV文件")
    
    if len(csv_files) == 0:
        print("❌ 错误: 数据目录中没有CSV文件")
        return False
    
    # 检查几个样本文件
    sample_files = csv_files[:3]  # 检查前3个文件
    print(f"\n📋 检查样本文件格式（前{len(sample_files)}个）:")
    
    for file_path in sample_files:
        print(f"\n  🔍 检查文件: {file_path.name}")
        
        try:
            # 只读取前几行检查格式
            sample = pd.read_csv(file_path, nrows=5)
            print(f"    ✅ 形状: {sample.shape}")
            print(f"    ✅ 列数: {len(sample.columns)}")
            print(f"    ✅ 第一列: {sample.columns[0]}")
            
            # 检查是否包含ID列
            first_col = sample.columns[0]
            if 'sequence' in first_col.lower() or 'seqn' in first_col.lower():
                print(f"    ✅ ID列识别正确")
            else:
                print(f"    ⚠️  警告: 第一列可能不是ID列")
            
            # 检查数据类型
            id_values = sample.iloc[:, 0].dropna()
            if len(id_values) > 0:
                print(f"    ✅ 样本ID值: {id_values.iloc[0]}")
            
        except Exception as e:
            print(f"    ❌ 错误: 无法读取文件 {file_path.name}: {e}")
            return False
    
    # 如果有变量字典，检查文件对应关系
    if variable_dict is not None:
        print(f"\n🔗 检查文件对应关系:")
        b_vars = variable_dict[variable_dict['category'].isin(['B', 'B&C'])]
        file_names = set(b_vars['file_name'].unique())
        existing_files = set([f.name for f in csv_files])
        
        missing_files = file_names - existing_files
        if missing_files:
            print(f"    ⚠️  警告: 变量字典中提到但文件不存在的文件:")
            for fname in sorted(list(missing_files))[:10]:  # 只显示前10个
                print(f"      - {fname}")
        else:
            print(f"    ✅ 所有必需的数据文件都存在")
    
    return True

def validate_sample_variable_loading(dict_path: str, data_dir: str) -> bool:
    """验证样本变量加载"""
    print(f"\n🧪 测试样本变量加载:")
    
    try:
        # 读取变量字典
        var_dict = pd.read_csv(dict_path)
        b_vars = var_dict[var_dict['category'].isin(['B', 'B&C'])]
        
        if len(b_vars) == 0:
            print("❌ 没有B/B&C类型变量可测试")
            return False
        
        # 测试加载前几个变量
        test_vars = b_vars.head(3)
        data_path = Path(data_dir)
        
        for idx, var_info in test_vars.iterrows():
            file_path = data_path / var_info['file_name']
            var_name = var_info['variable_name']
            
            print(f"  🔍 测试变量: {var_name}")
            print(f"    文件: {var_info['file_name']}")
            
            if not file_path.exists():
                print(f"    ❌ 文件不存在")
                continue
            
            try:
                # 读取表头
                header = pd.read_csv(file_path, nrows=0)
                columns = header.columns.tolist()
                
                if var_name not in columns:
                    print(f"    ❌ 变量不在文件中")
                    print(f"    可用列: {columns[:5]}...")  # 显示前5个列名
                    continue
                
                # 尝试加载该变量
                id_col = columns[0]
                data = pd.read_csv(file_path, usecols=[id_col, var_name], nrows=100)
                
                print(f"    ✅ 成功加载，形状: {data.shape}")
                print(f"    ✅ 缺失率: {data[var_name].isna().mean():.2%}")
                
                valid_values = data[var_name].dropna()
                if len(valid_values) > 0:
                    print(f"    ✅ 样本值: {valid_values.iloc[0]}")
                
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='验证NHANES数据格式')
    parser.add_argument('--variable_dict', type=str, 
                       default='all_nhanes_variables.csv',
                       help='变量字典文件路径')
    parser.add_argument('--data_dir', type=str,
                       default='NHANES_PROCESSED_CSV_merged_by_prefix',
                       help='数据文件目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 NHANES数据格式验证")
    print("="*60)
    
    all_passed = True
    
    # 1. 验证变量字典
    dict_valid = validate_variable_dictionary(args.variable_dict)
    all_passed = all_passed and dict_valid
    
    # 2. 验证数据目录
    var_dict = None
    if dict_valid:
        var_dict = pd.read_csv(args.variable_dict)
    
    dir_valid = validate_data_directory(args.data_dir, var_dict)
    all_passed = all_passed and dir_valid
    
    # 3. 测试样本变量加载
    if dict_valid and dir_valid:
        load_valid = validate_sample_variable_loading(args.variable_dict, args.data_dir)
        all_passed = all_passed and load_valid
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有验证通过！数据格式正确，可以运行主分析脚本。")
        print("\n🚀 运行主分析的命令:")
        print(f"python modularize_mediators_iteratively.py --variable_dict {args.variable_dict} --data_dir {args.data_dir}")
    else:
        print("❌ 验证失败！请修复上述问题后再运行主分析脚本。")
        sys.exit(1)
    
    print("="*60)

if __name__ == "__main__":
    main() 