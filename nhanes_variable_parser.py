#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHANES变量信息解析脚本
解析HTM文档文件，提取变量信息和数值映射表
创建结构化的数据标签与映射文件夹

文件夹结构：
- 主文件夹：统计表名称（如 ALQ_H）
- 子文件夹：变量名称（如 ALQ101）
- 每个变量文件夹包含：variable_info.json 和 variable_values.csv

作者: 自动生成
日期: 2024
"""

import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nhanes_variable_parsing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NHANESVariableParser:
    """NHANES变量信息解析器"""
    
    def __init__(self, base_dir: str = "NHANES_Data", output_dir: str = "NHANES_Variables"):
        """
        初始化解析器
        
        Args:
            base_dir: NHANES数据基础目录
            output_dir: 输出目录（变量信息存放位置）
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.htm_dir = self.base_dir / "Nhanes" / "文档文件"
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'processed_files': 0,
            'total_variables': 0,
            'variables_with_values': 0,
            'failed_files': [],
            'failed_variables': []
        }
    
    def extract_variable_info(self, variable_div) -> Optional[Dict]:
        """
        从variable div中提取变量信息
        
        Args:
            variable_div: BeautifulSoup的div元素
            
        Returns:
            变量信息字典或None
        """
        try:
            variable_info = {}
            
            # 提取Variable Name（使用更灵活的匹配）
            var_name_elem = variable_div.find('dt', string=re.compile(r'Variable Name:\s*', re.IGNORECASE))
            if not var_name_elem:
                var_name_elem = variable_div.find('dt', string='Variable Name: ')
            if var_name_elem and var_name_elem.find_next_sibling('dd'):
                variable_info['Variable_Name'] = var_name_elem.find_next_sibling('dd').get_text(strip=True)
            
            # 提取SAS Label
            label_elem = variable_div.find('dt', string=re.compile(r'SAS Label:\s*', re.IGNORECASE))
            if not label_elem:
                label_elem = variable_div.find('dt', string='SAS Label: ')
            if label_elem and label_elem.find_next_sibling('dd'):
                variable_info['Label'] = label_elem.find_next_sibling('dd').get_text(strip=True)
            
            # 提取English Text
            english_elem = variable_div.find('dt', string=re.compile(r'English Text:\s*', re.IGNORECASE))
            if not english_elem:
                english_elem = variable_div.find('dt', string='English Text: ')
            if english_elem and english_elem.find_next_sibling('dd'):
                variable_info['English_Text'] = english_elem.find_next_sibling('dd').get_text(strip=True)
            
            # 提取Target
            target_elem = variable_div.find('dt', string=re.compile(r'Target:\s*', re.IGNORECASE))
            if not target_elem:
                target_elem = variable_div.find('dt', string='Target: ')
            if target_elem and target_elem.find_next_sibling('dd'):
                variable_info['Target'] = target_elem.find_next_sibling('dd').get_text(strip=True)
            
            # 检查是否有必要的信息
            if 'Variable_Name' not in variable_info:
                return None
                
            return variable_info
            
        except Exception as e:
            logger.error(f"提取变量信息时出错: {str(e)}")
            return None
    
    def extract_values_table(self, variable_div) -> Optional[pd.DataFrame]:
        """
        从variable div中提取values表格
        
        Args:
            variable_div: BeautifulSoup的div元素
            
        Returns:
            pandas DataFrame或None
        """
        try:
            values_table = variable_div.find('table', class_='values')
            if not values_table:
                return None
            
            # 提取表头
            thead = values_table.find('thead')
            if not thead:
                return None
            
            headers = []
            header_row = thead.find('tr')
            if header_row:
                for th in header_row.find_all('th'):
                    headers.append(th.get_text(strip=True))
            
            if not headers:
                return None
            
            # 提取表格数据
            tbody = values_table.find('tbody')
            if not tbody:
                return None
            
            rows_data = []
            for row in tbody.find_all('tr'):
                row_data = []
                for td in row.find_all('td'):
                    # 获取单元格文本，处理可能的空值
                    cell_text = td.get_text(strip=True)
                    row_data.append(cell_text if cell_text else '')
                
                if row_data:  # 只添加非空行
                    rows_data.append(row_data)
            
            if not rows_data:
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(rows_data, columns=headers)
            return df
            
        except Exception as e:
            logger.error(f"提取values表格时出错: {str(e)}")
            return None
    
    def process_htm_file(self, htm_file_path: Path):
        """
        处理单个HTM文件
        
        Args:
            htm_file_path: HTM文件路径
        """
        try:
            logger.info(f"处理文件: {htm_file_path.name}")
            
            # 读取HTM文件
            with open(htm_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # 查找Codebook and Frequencies部分
            codebook_header = soup.find('h2', string=re.compile(r'Codebook and Frequencies', re.IGNORECASE))
            if not codebook_header:
                logger.warning(f"在 {htm_file_path.name} 中未找到 'Codebook and Frequencies' 部分")
                return
            
            # 获取HTM文件名（不含扩展名）
            htm_name = htm_file_path.stem
            
            # 查找所有variable的div（pagebreak类）
            variable_divs = soup.find_all('div', class_='pagebreak')
            
            if not variable_divs:
                logger.warning(f"在 {htm_file_path.name} 中未找到任何variable div")
                return
            
            file_variable_count = 0
            
            for variable_div in variable_divs:
                # 提取变量信息
                variable_info = self.extract_variable_info(variable_div)
                if not variable_info:
                    continue
                
                variable_name = variable_info.get('Variable_Name', f'unknown_{file_variable_count}')
                
                # 创建主文件夹（统计表）和变量子文件夹
                main_folder = self.output_dir / htm_name
                main_folder.mkdir(parents=True, exist_ok=True)
                variable_folder = main_folder / variable_name
                variable_folder.mkdir(parents=True, exist_ok=True)
                
                # 保存变量信息为JSON
                json_file = variable_folder / f"{variable_name}_info.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(variable_info, f, ensure_ascii=False, indent=2)
                
                # 提取并保存values表格
                values_df = self.extract_values_table(variable_div)
                if values_df is not None:
                    csv_file = variable_folder / f"{variable_name}_values.csv"
                    values_df.to_csv(csv_file, index=False, encoding='utf-8')
                    self.stats['variables_with_values'] += 1
                    logger.debug(f"保存了values表格: {csv_file}")
                
                file_variable_count += 1
                self.stats['total_variables'] += 1
                
                logger.info(f"成功处理变量: {variable_name}")
            
            self.stats['processed_files'] += 1
            logger.info(f"文件 {htm_file_path.name} 处理完成，共处理 {file_variable_count} 个变量")
            
        except Exception as e:
            logger.error(f"处理文件 {htm_file_path.name} 时出错: {str(e)}")
            self.stats['failed_files'].append(str(htm_file_path.name))
    
    def process_all_htm_files(self):
        """处理所有HTM文件"""
        logger.info("开始处理所有HTM文件...")
        
        if not self.htm_dir.exists():
            logger.error(f"HTM文档目录不存在: {self.htm_dir}")
            return
        
        # 获取所有HTM文件
        htm_files = list(self.htm_dir.glob("*.htm")) + list(self.htm_dir.glob("*.html"))
        
        if not htm_files:
            logger.error(f"在 {self.htm_dir} 中未找到任何HTM文件")
            return
        
        logger.info(f"找到 {len(htm_files)} 个HTM文件")
        
        # 处理每个文件
        for htm_file in htm_files:
            self.process_htm_file(htm_file)
    
    def create_summary_report(self):
        """创建处理摘要报告"""
        summary = {
            "处理时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "处理统计": {
                "已处理文件数": self.stats['processed_files'],
                "总变量数": self.stats['total_variables'],
                "带数值表的变量数": self.stats['variables_with_values'],
                "失败文件数": len(self.stats['failed_files'])
            },
            "失败文件列表": self.stats['failed_files'],
            "输出目录": str(self.output_dir.absolute()),
            "说明": {
                "文件夹结构": "统计表（如ALQ_H）为主文件夹，每个变量（如ALQ101）为子文件夹",
                "JSON文件": "包含变量的基本信息",
                "CSV文件": "包含变量的数值对应表（如果有的话）"
            }
        }
        
        # 保存摘要
        summary_path = self.output_dir / "parsing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理摘要已保存: {summary_path}")
        return summary
    
    def run(self):
        """运行解析流程"""
        logger.info("="*60)
        logger.info("开始NHANES变量信息解析...")
        logger.info("="*60)
        
        try:
            # 处理所有HTM文件
            self.process_all_htm_files()
            
            # 创建摘要报告
            summary = self.create_summary_report()
            
            # 输出最终统计
            logger.info("="*60)
            logger.info("NHANES变量解析完成!")
            logger.info("="*60)
            logger.info(f"已处理文件数: {self.stats['processed_files']}")
            logger.info(f"总变量数: {self.stats['total_variables']}")
            logger.info(f"带数值表的变量数: {self.stats['variables_with_values']}")
            logger.info(f"失败文件数: {len(self.stats['failed_files'])}")
            
            if self.stats['total_variables'] > 0:
                values_rate = (self.stats['variables_with_values'] / self.stats['total_variables']) * 100
                logger.info(f"数值表覆盖率: {values_rate:.1f}%")
            
            if self.stats['failed_files']:
                logger.warning(f"失败的文件: {', '.join(self.stats['failed_files'])}")
            
            logger.info(f"输出目录: {self.output_dir.absolute()}")
            
        except Exception as e:
            logger.error(f"运行过程中发生错误: {str(e)}")

def main():
    """主函数"""
    print("NHANES变量信息解析脚本")
    print("="*50)
    
    # 创建解析器实例
    parser = NHANESVariableParser()
    
    # 开始解析
    try:
        parser.run()
    except KeyboardInterrupt:
        logger.info("\n用户中断解析")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
    
    print("\n程序执行完成")

if __name__ == "__main__":
    main() 