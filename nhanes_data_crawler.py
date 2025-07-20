#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHANES数据抓取脚本
从CDC官方网站动态获取并下载NHANES数据
先获取所有可用的数据文件链接，然后按年份组织下载

作者: 自动生成
日期: 2024
"""

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import zipfile
import io

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nhanes_download.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NHANESCrawler:
    """NHANES数据爬虫类 - 动态获取数据链接版本"""
    
    def __init__(self, base_dir: str = "NHANES_Data"):
        """
        初始化爬虫
        
        Args:
            base_dir: 数据保存的基础目录
        """
        self.base_dir = Path(base_dir)
        
        # 定义NHANES网站的组件分类
        self.components = ["Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire"]
        self.base_url_variablelist = "https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component="
        self.base_url_datapage = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component="
        
        # 初始化数据存储
        self.all_variables = None
        self.all_datafiles = None
        self.xpt_links = []
        
        # 初始化下载统计
        self.download_stats = {
            'total_attempts': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_existing': 0,
            'failed_files': []
        }
        
        # 创建基础目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_variable_lists(self):
        """获取所有组件的变量列表"""
        logger.info("开始获取NHANES变量列表...")
        
        tables = []
        for component in self.components:
            url = self.base_url_variablelist + component
            logger.info(f"正在获取 {component} 组件的变量列表...")
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                table = soup.find('table')
                
                if table:
                    df = pd.read_html(str(table))[0]
                    df['Component'] = component  # 添加组件标识
                    tables.append(df)
                    logger.info(f"{component} 组件找到 {len(df)} 个变量")
                else:
                    logger.warning(f"{component} 组件未找到表格")
                    
            except Exception as e:
                logger.error(f"获取 {component} 组件变量列表失败: {str(e)}")
                continue
            
            time.sleep(1)  # 避免请求过于频繁
        
        if tables:
            self.all_variables = pd.concat(tables, ignore_index=True)
            logger.info(f"总共获取到 {len(self.all_variables)} 个变量")
            
            # 保存变量列表
            self.all_variables.to_pickle(self.base_dir / "Nhanes_Variablelist.pkl")
            self.all_variables.to_excel(self.base_dir / "Nhanes_Variablelist.xlsx", index=False)
            logger.info("变量列表已保存")
        else:
            logger.error("未能获取到任何变量列表")
    
    def get_data_file_tables(self):
        """获取所有组件的数据文件表格"""
        logger.info("开始获取NHANES数据文件表格...")
        
        tables_data = []
        for component in self.components:
            url = self.base_url_datapage + component
            logger.info(f"正在获取 {component} 组件的数据文件表格...")
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                table = soup.find('table')
                
                if table:
                    df = pd.read_html(str(table))[0]
                    df['Component'] = component  # 添加组件标识
                    tables_data.append(df)
                    logger.info(f"{component} 组件找到 {len(df)} 个数据文件")
                else:
                    logger.warning(f"{component} 组件未找到数据文件表格")
                    
            except Exception as e:
                logger.error(f"获取 {component} 组件数据文件表格失败: {str(e)}")
                continue
            
            time.sleep(1)  # 避免请求过于频繁
        
        if tables_data:
            self.all_datafiles = pd.concat(tables_data, ignore_index=True)
            logger.info(f"总共获取到 {len(self.all_datafiles)} 个数据文件")
            
            # 保存数据文件表格
            self.all_datafiles.to_pickle(self.base_dir / "Data_File_Table.pkl")
            self.all_datafiles.to_excel(self.base_dir / "Data_File_Table.xlsx", index=False)
            logger.info("数据文件表格已保存")
        else:
            logger.error("未能获取到任何数据文件表格")
    
    def extract_xpt_links(self):
        """提取所有.XPT文件的下载链接"""
        logger.info("开始提取.XPT文件下载链接...")
        
        all_hrefs = []
        for component in self.components:
            url = self.base_url_datapage + component
            logger.info(f"正在从 {component} 组件提取链接...")
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                component_hrefs = [link['href'] for link in links]
                all_hrefs.extend(component_hrefs)
                logger.info(f"从 {component} 组件提取到 {len(component_hrefs)} 个链接")
                
            except Exception as e:
                logger.error(f"从 {component} 组件提取链接失败: {str(e)}")
                continue
            
            time.sleep(1)  # 避免请求过于频繁
        
        # 过滤出.XPT文件链接
        xpt_hrefs = [href for href in all_hrefs if ".XPT" in href.upper()]
        
        # 添加完整URL前缀
        self.xpt_links = ["https://wwwn.cdc.gov" + link for link in xpt_hrefs]
        
        logger.info(f"总共找到 {len(self.xpt_links)} 个.XPT文件链接")
        
        if self.xpt_links:
            # 保存链接列表
            df_links = pd.DataFrame(self.xpt_links, columns=["Links"])
            df_links.to_excel(self.base_dir / "Nhanes_Data_href.xlsx", index=False)
            df_links.to_pickle(self.base_dir / "Nhanes_Data_href.pkl")
            
            # 显示前几个链接作为示例
            logger.info("前几个.XPT文件链接示例:")
            for i, link in enumerate(self.xpt_links[:6]):
                logger.info(f"  {i+1}. {link}")
            
            logger.info("链接列表已保存")
        else:
            logger.error("未找到任何.XPT文件链接")
    
    def organize_links_by_year(self):
        """按年份组织链接"""
        if not self.xpt_links:
            logger.error("没有可用的链接进行组织")
            return {}
        
        links_by_year = {}
        for link in self.xpt_links:
            try:
                # 从链接中提取年份信息，例如: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT
                parts = link.split('/')
                if len(parts) >= 6:
                    year_cycle = parts[5]  # 例如 "2017-2018"
                    if year_cycle not in links_by_year:
                        links_by_year[year_cycle] = []
                    links_by_year[year_cycle].append(link)
            except Exception as e:
                logger.warning(f"无法解析链接年份: {link} - {str(e)}")
                continue
        
        logger.info(f"链接已按 {len(links_by_year)} 个年份周期组织:")
        for year, links in links_by_year.items():
            logger.info(f"  {year}: {len(links)} 个文件")
        
        return links_by_year
    
    def download_file(self, link: str, save_dir: Path) -> bool:
        """下载单个文件"""
        try:
            filename = link.split('/')[-1]
            file_path = save_dir / filename
            
            # 检查文件是否已存在
            if file_path.exists():
                logger.info(f"文件已存在，跳过: {filename}")
                self.download_stats['skipped_existing'] += 1
                return True
            
            logger.info(f"正在下载: {filename}")
            response = requests.get(link, stream=True, timeout=60)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 验证文件大小
                file_size = file_path.stat().st_size
                if file_size < 1000:  # 小于1KB可能是错误页面
                    logger.warning(f"下载的文件太小，可能是错误页面: {filename} ({file_size} bytes)")
                    file_path.unlink()  # 删除错误文件
                    return False
                
                logger.info(f"下载完成: {filename} ({file_size} bytes)")
                self.download_stats['successful_downloads'] += 1
                return True
            else:
                logger.error(f"下载失败: {filename} - HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"下载 {link} 时发生错误: {str(e)}")
            return False
    
    def download_all_files(self):
        """下载所有文件，按年份组织"""
        if not self.xpt_links:
            logger.error("没有可用的下载链接")
            return
        
        logger.info("开始下载所有NHANES数据文件...")
        
        # 按年份组织链接
        links_by_year = self.organize_links_by_year()
        
        if not links_by_year:
            logger.error("无法按年份组织链接")
            return
        
        # 按年份下载文件
        for year_cycle, links in links_by_year.items():
            logger.info(f"\n开始下载 {year_cycle} 年份的数据...")
            
            # 创建年份目录
            year_dir = self.base_dir / year_cycle
            year_dir.mkdir(parents=True, exist_ok=True)
            
            # 下载该年份的所有文件
            for link in links:
                self.download_stats['total_attempts'] += 1
                
                success = self.download_file(link, year_dir)
                if not success:
                    self.download_stats['failed_downloads'] += 1
                    self.download_stats['failed_files'].append({
                        'link': link,
                        'year': year_cycle,
                        'filename': link.split('/')[-1]
                    })
                
                time.sleep(0.5)  # 避免请求过于频繁
            
            logger.info(f"{year_cycle} 年份下载完成")
    
    def create_summary_report(self):
        """创建下载摘要报告"""
        summary = {
            "数据来源": "NHANES (National Health and Nutrition Examination Survey)",
            "下载时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "获取方法": "动态从CDC网站抓取最新的数据文件链接",
            "下载统计": {
                "总尝试下载": self.download_stats['total_attempts'],
                "成功下载": self.download_stats['successful_downloads'],
                "失败下载": self.download_stats['failed_downloads'],
                "跳过已存在": self.download_stats['skipped_existing']
            },
            "失败文件": self.download_stats['failed_files'],
            "数据组织": "按调查周期（年份）分组存储",
            "文件格式": "SAS Transport格式 (.XPT)"
        }
        
        if self.download_stats['total_attempts'] > 0:
            success_rate = (self.download_stats['successful_downloads'] / self.download_stats['total_attempts']) * 100
            summary["下载统计"]["成功率"] = f"{success_rate:.1f}%"
        
        # 保存摘要
        import json
        summary_path = self.base_dir / "下载摘要.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"下载摘要已保存: {summary_path}")
        return summary
    


    
    def run(self):
        """运行数据获取和下载流程"""
        logger.info("开始NHANES数据获取流程...")
        
        try:
            # 第一步：获取变量列表
            self.get_variable_lists()
            
            # 第二步：获取数据文件表格
            self.get_data_file_tables()
            
            # 第三步：提取所有.XPT文件链接
            self.extract_xpt_links()
            
            # 显示获取到的信息摘要
            logger.info(f"\n{'='*60}")
            logger.info("数据获取完成摘要:")
            logger.info(f"{'='*60}")
            
            if self.all_variables is not None:
                logger.info(f"变量列表: {len(self.all_variables)} 个变量")
            
            if self.all_datafiles is not None:
                logger.info(f"数据文件: {len(self.all_datafiles)} 个数据文件")
            
            logger.info(f".XPT下载链接: {len(self.xpt_links)} 个文件")
            
            # 第四步：询问是否继续下载
            if self.xpt_links:
                logger.info(f"\n发现 {len(self.xpt_links)} 个可下载的数据文件")
                logger.info("链接已保存到 'Nhanes_Data_href.xlsx' 文件中")
                
                # 按年份统计
                links_by_year = self.organize_links_by_year()
                
                # 开始下载所有文件
                logger.info("\n开始下载所有数据文件...")
                self.download_all_files()
                
                # 创建下载摘要
                self.create_summary_report()
                
                # 输出最终统计
                logger.info(f"\n{'='*60}")
                logger.info("NHANES数据下载完成!")
                logger.info(f"{'='*60}")
                logger.info(f"总计尝试下载: {self.download_stats['total_attempts']} 个文件")
                logger.info(f"成功下载: {self.download_stats['successful_downloads']} 个文件")
                logger.info(f"跳过已存在: {self.download_stats['skipped_existing']} 个文件")
                logger.info(f"失败下载: {self.download_stats['failed_downloads']} 个文件")
                
                if self.download_stats['total_attempts'] > 0:
                    success_rate = (self.download_stats['successful_downloads'] / self.download_stats['total_attempts']) * 100
                    logger.info(f"总体成功率: {success_rate:.1f}%")
                
                if self.download_stats['failed_files']:
                    logger.info(f"\n失败文件数量: {len(self.download_stats['failed_files'])}")
                    logger.info("详细失败信息请查看 '下载摘要.json' 文件")
                
                logger.info(f"\n数据保存位置: {self.base_dir.absolute()}")
                logger.info("数据已按调查周期（年份）分组，便于后续分析使用")
            else:
                logger.error("未找到任何可下载的数据文件链接")
        
        except Exception as e:
            logger.error(f"运行过程中发生错误: {str(e)}")
            # 即使出错也尝试保存已获取的信息
            try:
                self.create_summary_report()
            except:
                pass

def main():
    """主函数"""
    print("NHANES数据抓取脚本")
    print("="*50)
    
    # 创建爬虫实例
    crawler = NHANESCrawler()
    
    # 开始下载
    try:
        crawler.run()
    except KeyboardInterrupt:
        logger.info("\n用户中断下载")
        # 即使中断也保存已有的统计信息
        crawler.save_download_summary()
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        # 保存错误前的统计信息
        crawler.save_download_summary()
    
    print("\n程序执行完成")

if __name__ == "__main__":
    main() 