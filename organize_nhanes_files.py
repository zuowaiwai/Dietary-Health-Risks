#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHANES 数据文件整理和预处理脚本
功能：
1. 将散乱的NHANES .xpt文件按照数据类别和调查周期进行分类整理
2. 将XPT文件转换为CSV格式
3. 标准化关键列名
4. 合并年度数据为单个文件
作者：AI助手
日期：2024
"""

import os
import shutil
import re
import pandas as pd
import logging
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ================== 配置区域 ==================
# 源目录 - 包含所有未整理的.xpt文件
SOURCE_DIR = "NHANES_Organized/_Uncategorized"

# 目标目录 - 整理后的文件将存放在这里
DEST_DIR = "NHANES_Organized"

# CSV输出目录 - 转换后的CSV文件存放目录
CSV_OUTPUT_DIR = "NHANES_PROCESSED_CSV"

# CSV原始输出目录 - 不含标签增强的CSV文件存放目录
CSV_OUTPUT_DIR_RAW = "Nhanes_processed_csv_raw"

# 变量描述目录 - 包含变量标签和数值标签信息
VARIABLES_DIR = "NHANES_Variables"

# 日志文件路径
LOG_FILE = "nhanes_preprocessing.log"

# ================== 日志配置 ==================

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)

# ================== 映射字典 ==================

# 年份映射：从文件名后缀到调查周期
YEAR_MAP = {
    '_B': '2001-2002',
    '_C': '2003-2004', 
    '_D': '2005-2006',
    '_E': '2007-2008',
    '_F': '2009-2010',
    '_G': '2011-2012',
    '_H': '2013-2014',
    '_I': '2015-2016',
    '_J': '2017-2018',
    '_L': '2021-2022',
    'P_': '2019-2020',  # P_前缀表示疫情期间数据
    '': '1999-2000'     # 无后缀的文件通常是1999-2000年
}

# 标准列名映射：用于统一关键列名
COLUMN_RENAME_MAP = {
    'SEQN': 'RespondentSequenceNumber',
    'SDDSRVYR': 'DataReleaseCycle',
    'RIDSTATR': 'InterviewStatus',
    'RIAGENDR': 'Gender',
    'RIDAGEYR': 'AgeInYearsAtScreening',
    'RIDAGEMN': 'AgeInMonthsAtScreening',
    'RIDRETH1': 'RaceEthnicityRecode1',
    'RIDRETH3': 'RaceEthnicityRecode3',
    'DMDBORN4': 'CountryOfBirth',
    'WTINT2YR': 'FullSample2YearInterviewWeight',
    'WTMEC2YR': 'FullSample2YearMECExamWeight',
    # 可以根据需要继续添加
}

# 类别映射：从文件名前缀到类别路径
CATEGORY_MAP = {
    # A. 问卷调查数据
    # 1. 人口统计学信息
    'DEMO': ('A_问卷调查数据', '1_人口统计学信息'),
    
    # 2. 膳食数据
    'DR1TOT': ('A_问卷调查数据', '2_膳食数据_总摄入量'),
    'DR2TOT': ('A_问卷调查数据', '2_膳食数据_总摄入量'),
    'DR1IFF': ('A_问卷调查数据', '2_膳食数据_单个食物'),
    'DR2IFF': ('A_问卷调查数据', '2_膳食数据_单个食物'),
    'DRXFCD': ('A_问卷调查数据', '2_膳食数据_食物编码'),
    'DRXFMT': ('A_问卷调查数据', '2_膳食数据_食物编码'),
    'DRXMCD': ('A_问卷调查数据', '2_膳食数据_食物编码'),
    'DRXTOT': ('A_问卷调查数据', '2_膳食数据_总摄入量'),
    'DRXIFF': ('A_问卷调查数据', '2_膳食数据_单个食物'),
    'DBQ': ('A_问卷调查数据', '2_膳食行为问卷'),
    'FFQDC': ('A_问卷调查数据', '2_食物频率问卷'),
    'FFQRAW': ('A_问卷调查数据', '2_食物频率问卷'),
    
    # 3. 膳食补充剂
    'DSQTOT': ('A_问卷调查数据', '3_膳食补充剂_总量'),
    'DSQIDS': ('A_问卷调查数据', '3_膳食补充剂_明细'),
    'DS1TOT': ('A_问卷调查数据', '3_膳食补充剂_第一天'),
    'DS1IDS': ('A_问卷调查数据', '3_膳食补充剂_第一天'),
    'DS2TOT': ('A_问卷调查数据', '3_膳食补充剂_第二天'),
    'DS2IDS': ('A_问卷调查数据', '3_膳食补充剂_第二天'),
    'DSQ1': ('A_问卷调查数据', '3_膳食补充剂_问卷1'),
    'DSQ2': ('A_问卷调查数据', '3_膳食补充剂_问卷2'),
    'DSBI': ('A_问卷调查数据', '3_膳食补充剂_品牌信息'),
    'DSII': ('A_问卷调查数据', '3_膳食补充剂_成分信息'),
    'DSPI': ('A_问卷调查数据', '3_膳食补充剂_产品信息'),
    'DSQFILE1': ('A_问卷调查数据', '3_膳食补充剂_文件1'),
    'DSQFILE2': ('A_问卷调查数据', '3_膳食补充剂_文件2'),
    
    # 4. 生活方式
    'PAQ': ('A_问卷调查数据', '4_生活方式_体力活动'),
    'PAQIAF': ('A_问卷调查数据', '4_生活方式_体力活动_个人'),
    'SMQ': ('A_问卷调查数据', '4_生活方式_吸烟_个人'),
    'SMQFAM': ('A_问卷调查数据', '4_生活方式_吸烟_家庭'),
    'SMQRTU': ('A_问卷调查数据', '4_生活方式_吸烟_近期使用'),
    'SMQSHS': ('A_问卷调查数据', '4_生活方式_二手烟'),
    'SMQMEC': ('A_问卷调查数据', '4_生活方式_吸烟_MEC'),
    'ALQ': ('A_问卷调查数据', '4_生活方式_饮酒'),
    'ALQY': ('A_问卷调查数据', '4_生活方式_饮酒_青年'),
    'SLQ': ('A_问卷调查数据', '4_生活方式_睡眠'),
    'SXQ': ('A_问卷调查数据', '4_生活方式_性行为'),
    'DUQ': ('A_问卷调查数据', '4_生活方式_药物使用'),
    
    # 5. 疾病史和健康状况
    'MCQ': ('A_问卷调查数据', '5_疾病史_一般'),
    'DIQ': ('A_问卷调查数据', '5_疾病史_糖尿病'),
    'BPQ': ('A_问卷调查数据', '5_疾病史_血压'),
    'CDQ': ('A_问卷调查数据', '5_疾病史_心血管'),
    'CKQ': ('A_问卷调查数据', '5_疾病史_慢性肾病'),
    'KIQ_U': ('A_问卷调查数据', '5_疾病史_肾脏_泌尿'),
    'HSQ': ('A_问卷调查数据', '5_健康状况_当前'),
    'HUQ': ('A_问卷调查数据', '5_健康状况_医疗利用'),
    'HIQ': ('A_问卷调查数据', '5_健康状况_医疗保险'),
    'HEQ': ('A_问卷调查数据', '5_健康状况_听力'),
    'CFQ': ('A_问卷调查数据', '5_认知功能'),
    'DLQ': ('A_问卷调查数据', '5_物理功能_残疾'),
    'PFQ': ('A_问卷调查数据', '5_物理功能'),
    
    # 6. 药物和保健品
    'RXQ_RX': ('A_问卷调查数据', '6_处方药物'),
    'RXQANA': ('A_问卷调查数据', '6_处方药物_分析'),
    'RXQ_DRUG': ('A_问卷调查数据', '6_处方药物_详细'),
    
    # 7. 女性健康
    'RHQ': ('A_问卷调查数据', '7_女性健康_生殖'),
    'OHQ': ('A_问卷调查数据', '7_女性健康_口服避孕药'),
    
    # 8. 免疫和疫苗
    'IMQ': ('A_问卷调查数据', '8_免疫_疫苗史'),
    
    # 9. 职业和环境
    'OCQ': ('A_问卷调查数据', '9_职业_工作'),
    'OPD': ('A_问卷调查数据', '9_职业_农药'),
    
    # 10. 住房和环境
    'HOQ': ('A_问卷调查数据', '10_住房_特征'),
    'DEQ': ('A_问卷调查数据', '10_皮肤病_防晒'),
    
    # 11. 精神健康
    'DPQ': ('A_问卷调查数据', '11_精神健康_抑郁'),
    'CIQDEP': ('A_问卷调查数据', '11_精神健康_抑郁_CIDI'),
    'CIQGAD': ('A_问卷调查数据', '11_精神健康_焦虑_CIDI'),
    'CIQPAN': ('A_问卷调查数据', '11_精神健康_惊恐_CIDI'),
    'CIQPANIC': ('A_问卷调查数据', '11_精神健康_惊恐_CIDI'),
    
    # 12. 体重和节食
    'WHQ': ('A_问卷调查数据', '12_体重_历史'),
    'WHQMEC': ('A_问卷调查数据', '12_体重_MEC'),
    'WDQ': ('A_问卷调查数据', '12_体重_节食'),
    
    # 13. 其他问卷
    'AUQ': ('A_问卷调查数据', '13_其他_听力问卷'),
    'VTQ': ('A_问卷调查数据', '13_其他_挥发性毒物'),
    'VIQ': ('A_问卷调查数据', '13_其他_视力_眼科'),
    'TBQ': ('A_问卷调查数据', '13_其他_结核病'),
    'CSQ': ('A_问卷调查数据', '13_其他_医疗条件筛查'),
    'SSQ': ('A_问卷调查数据', '13_其他_社会支持'),
    'BHQ': ('A_问卷调查数据', '13_其他_行为健康'),
    'CBQ': ('A_问卷调查数据', '13_其他_消费行为'),
    'CBQPFA': ('A_问卷调查数据', '13_其他_消费行为_PFAS'),
    'CBQPFC': ('A_问卷调查数据', '13_其他_消费行为_PFAS'),
    'PUQMEC': ('A_问卷调查数据', '13_其他_杀虫剂'),
    'PUQ': ('A_问卷调查数据', '13_其他_杀虫剂_详细'),
    'ACQ': ('A_问卷调查数据', '13_其他_住宿条件'),
    'BAQ': ('A_问卷调查数据', '13_其他_文身'),
    'ECQ': ('A_问卷调查数据', '13_其他_幼儿护理'),
    'AQQ': ('A_问卷调查数据', '13_其他_空气质量'),
    'ARQ': ('A_问卷调查数据', '13_其他_关节炎'),
    'DTQ': ('A_问卷调查数据', '13_其他_牙科'),
    'FNQ': ('A_问卷调查数据', '13_其他_食品安全'),
    'FSQ': ('A_问卷调查数据', '13_其他_食品安全_详细'),
    'RDQ': ('A_问卷调查数据', '13_其他_呼吸道健康'),
    'OSQ': ('A_问卷调查数据', '13_其他_职业健康'),
    'MPQ': ('A_问卷调查数据', '13_其他_精神健康_详细'),
    'INQ': ('A_问卷调查数据', '13_其他_收入问卷'),
    'HCQ': ('A_问卷调查数据', '13_其他_丙肝问卷'),
    'PSQ': ('A_问卷调查数据', '13_其他_前列腺症状'),
    'KIQ_P': ('A_问卷调查数据', '13_其他_儿科肾脏'),
    'KIQ': ('A_问卷调查数据', '13_其他_肾脏疾病'),
    'RXQASA': ('A_问卷调查数据', '13_其他_阿司匹林使用'),
    'RXQ_ANA': ('A_问卷调查数据', '13_其他_药物分析'),
    'AGQ': ('A_问卷调查数据', '13_其他_哮喘问卷'),
    
    # B. 体格检查数据
    # 1. 人体测量
    'BMX': ('B_体格检查数据', '1_人体测量_身体'),
    
    # 2. 血压
    'BPX': ('B_体格检查数据', '2_血压_水银'),
    'BPXO': ('B_体格检查数据', '2_血压_示波器'),
    
    # 3. 口腔健康
    'OHX': ('B_体格检查数据', '3_口腔健康_检查'),
    'OHXDEN': ('B_体格检查数据', '3_口腔健康_牙齿'),
    'OHXREF': ('B_体格检查数据', '3_口腔健康_推荐'),
    'OHXPER': ('B_体格检查数据', '3_口腔健康_牙周'),
    
    # 4. 骨密度
    'DXX': ('B_体格检查数据', '4_骨密度_全身'),
    'DXXAG': ('B_体格检查数据', '4_骨密度_年龄调整'),
    'DXXFEM': ('B_体格检查数据', '4_骨密度_股骨'),
    'DXXSPN': ('B_体格检查数据', '4_骨密度_脊柱'),
    'DXXFRX': ('B_体格检查数据', '4_骨密度_骨折史'),
    'DXXL1': ('B_体格检查数据', '4_骨密度_L1椎体'),
    'DXXL2': ('B_体格检查数据', '4_骨密度_L2椎体'),
    'DXXL3': ('B_体格检查数据', '4_骨密度_L3椎体'),
    'DXXL4': ('B_体格检查数据', '4_骨密度_L4椎体'),
    'DXXT5': ('B_体格检查数据', '4_骨密度_T5椎体'),
    'DXXT6': ('B_体格检查数据', '4_骨密度_T6椎体'),
    'DXXT7': ('B_体格检查数据', '4_骨密度_T7椎体'),
    'DXXT8': ('B_体格检查数据', '4_骨密度_T8椎体'),
    'DXXT10': ('B_体格检查数据', '4_骨密度_T10椎体'),
    'DXXT11': ('B_体格检查数据', '4_骨密度_T11椎体'),
    'DXXT12': ('B_体格检查数据', '4_骨密度_T12椎体'),
    'DXXAAC': ('B_体格检查数据', '4_骨密度_腹主动脉钙化'),
    'DXXVFA': ('B_体格检查数据', '4_骨密度_椎体骨折评估'),
    'DXX_2': ('B_体格检查数据', '4_骨密度_附加测量'),
    
    # 5. 听力测试
    'AUX': ('B_体格检查数据', '5_听力_纯音'),
    'AUXAR': ('B_体格检查数据', '5_听力_声反射'),
    'AUXTYM': ('B_体格检查数据', '5_听力_鼓膜'),
    'AUXWBR': ('B_体格检查数据', '5_听力_宽带反射'),
    
    # 6. 皮肤病学
    'DEX': ('B_体格检查数据', '6_皮肤_检查'),
    
    # 7. 心血管健康
    'CVX': ('B_体格检查数据', '7_心血管_检查'),
    
    # 8. 眼科检查
    'VIX': ('B_体格检查数据', '8_眼科_视力检查'),
    'OPXRET': ('B_体格检查数据', '8_眼科_视网膜摄影'),
    'OPXFDT': ('B_体格检查数据', '8_眼科_眼底检查'),
    
    # 9. 体力活动监测
    'PAXHR': ('B_体格检查数据', '9_体力活动_心率'),
    'PAXHD': ('B_体格检查数据', '9_体力活动_头部位置'),
    'PAXDAY': ('B_体格检查数据', '9_体力活动_日间数据'),
    
    # 10. 其他检查
    'BAX': ('B_体格检查数据', '10_其他_文身'),
    'BIX': ('B_体格检查数据', '10_其他_生物阻抗'),
    'ARX': ('B_体格检查数据', '10_其他_关节炎'),
    'ENX': ('B_体格检查数据', '10_其他_环境氮氧化物'),
    'CSX': ('B_体格检查数据', '10_其他_医疗条件筛查'),
    'SPX': ('B_体格检查数据', '10_其他_肺活量'),
    'MSX': ('B_体格检查数据', '10_其他_精神状态'),
    
    # C. 实验室数据
    # 1. 常规生化 - 血脂
    'TCHOL': ('C_实验室数据', '1_常规生化_血脂_总胆固醇'),
    'HDL': ('C_实验室数据', '1_常规生化_血脂_HDL'),
    'LDL': ('C_实验室数据', '1_常规生化_血脂_LDL'),
    'TRIGLY': ('C_实验室数据', '1_常规生化_血脂_甘油三酯'),
    'APOB': ('C_实验室数据', '1_常规生化_血脂_载脂蛋白B'),
    
    # 1. 常规生化 - 血糖
    'GLU': ('C_实验室数据', '1_常规生化_血糖_葡萄糖'),
    'GHB': ('C_实验室数据', '1_常规生化_血糖_糖化血红蛋白'),
    'INS': ('C_实验室数据', '1_常规生化_血糖_胰岛素'),
    'OGTT': ('C_实验室数据', '1_常规生化_血糖_糖耐量'),
    
    # 1. 常规生化 - 其他
    'BIOPRO': ('C_实验室数据', '1_常规生化_生化指标'),
    'THYROD': ('C_实验室数据', '1_常规生化_甲状腺功能'),
    'PSA': ('C_实验室数据', '1_常规生化_前列腺特异性抗原'),
    'PTH': ('C_实验室数据', '1_常规生化_甲状旁腺激素'),
    'PP': ('C_实验室数据', '1_常规生化_蛋白质'),
    'PH': ('C_实验室数据', '1_常规生化_酸碱度'),
    'HCY': ('C_实验室数据', '1_常规生化_同型半胱氨酸'),
    'MMA': ('C_实验室数据', '1_常规生化_甲基丙二酸'),
    'IHG': ('C_实验室数据', '1_常规生化_无机汞'),
    'MGX': ('C_实验室数据', '1_常规生化_镁'),
    'FASTQX': ('C_实验室数据', '1_常规生化_禁食问卷'),
    
    # 2. 血液学
    'CBC': ('C_实验室数据', '2_血液学_全血细胞计数'),
    
    # 3. 营养状况
    'FERTIN': ('C_实验室数据', '3_营养状况_铁蛋白'),
    'FETIB': ('C_实验室数据', '3_营养状况_铁结合力'),
    'TFR': ('C_实验室数据', '3_营养状况_转铁蛋白受体'),
    'VID': ('C_实验室数据', '3_营养状况_维生素D'),
    'VIT': ('C_实验室数据', '3_营养状况_维生素'),
    'VIC': ('C_实验室数据', '3_营养状况_维生素C'),
    'FOLATE': ('C_实验室数据', '3_营养状况_叶酸'),
    'FOLFMS': ('C_实验室数据', '3_营养状况_叶酸代谢物'),
    'B12': ('C_实验室数据', '3_营养状况_维生素B12'),
    'CARB': ('C_实验室数据', '3_营养状况_类胡萝卜素'),
    'TFA': ('C_实验室数据', '3_营养状况_反式脂肪酸'),
    'PHYTO': ('C_实验室数据', '3_营养状况_植物雌激素'),
    'PERNT': ('C_实验室数据', '3_营养状况_全氟和多氟化合物'),
    'PERNTS': ('C_实验室数据', '3_营养状况_全氟和多氟化合物_血清'),
    'CAFE': ('C_实验室数据', '3_营养状况_咖啡因'),
    'EPH': ('C_实验室数据', '3_营养状况_麻黄碱'),
    'EPHPP': ('C_实验室数据', '3_营养状况_麻黄碱代谢物'),
    'EPP': ('C_实验室数据', '3_营养状况_卟啉'),
    
    # 4. 炎症标志物
    'CRP': ('C_实验室数据', '4_炎症_C反应蛋白'),
    'HSCRP': ('C_实验室数据', '4_炎症_高敏C反应蛋白'),
    
    # 5. 肝肾功能
    'ALB_CR': ('C_实验室数据', '5_肝肾功能_白蛋白肌酐'),
    'UCFLOW': ('C_实验室数据', '5_肝肾功能_尿流率'),
    'UCPREG': ('C_实验室数据', '5_肝肾功能_尿妊娠'),
    'UCOT': ('C_实验室数据', '5_肝肾功能_尿肌酐'),
    'UC': ('C_实验室数据', '5_肝肾功能_尿肌酐_简版'),
    'UCM': ('C_实验室数据', '5_肝肾功能_尿肌酐_微量白蛋白'),
    'UCOSMO': ('C_实验室数据', '5_肝肾功能_尿渗透压'),
    'UTASS': ('C_实验室数据', '5_肝肾功能_尿砷'),
    'UAS': ('C_实验室数据', '5_肝肾功能_尿砷_详细'),
    'UTAS': ('C_实验室数据', '5_肝肾功能_尿总砷'),
    'UIO': ('C_实验室数据', '5_肝肾功能_尿碘'),
    'UM': ('C_实验室数据', '5_肝肾功能_尿汞'),
    'UMS': ('C_实验室数据', '5_肝肾功能_尿汞_汞代谢物'),
    'UAM': ('C_实验室数据', '5_肝肾功能_尿砷代谢物'),
    'UADM': ('C_实验室数据', '5_肝肾功能_尿砷二甲基酸'),
    'UPP': ('C_实验室数据', '5_肝肾功能_尿卟啉原'),
    'UNI': ('C_实验室数据', '5_肝肾功能_尿镍'),
    
    # 6. 感染性疾病
    'HEPB_S': ('C_实验室数据', '6_感染_乙肝表面抗原'),
    'HEPA': ('C_实验室数据', '6_感染_甲肝'),
    'HEPBD': ('C_实验室数据', '6_感染_乙肝DNA'),
    'HEPC': ('C_实验室数据', '6_感染_丙肝'),
    'HEPD': ('C_实验室数据', '6_感染_丁肝'),
    'HEPE': ('C_实验室数据', '6_感染_戊肝'),
    'VNA': ('C_实验室数据', '6_感染_病毒性肝炎'),
    'VNAS': ('C_实验室数据', '6_感染_病毒性肝炎_血清'),
    'HIV': ('C_实验室数据', '6_感染_HIV'),
    'HSV': ('C_实验室数据', '6_感染_疱疹病毒'),
    'CMV': ('C_实验室数据', '6_感染_巨细胞病毒'),
    'CHLMDA': ('C_实验室数据', '6_感染_衣原体'),
    'TRICH': ('C_实验室数据', '6_感染_滴虫'),
    'TGEMA': ('C_实验室数据', '6_感染_弓形虫'),
    'TB': ('C_实验室数据', '6_感染_结核病'),
    'TST': ('C_实验室数据', '6_感染_结核菌素皮试'),
    'TELO': ('C_实验室数据', '6_感染_端粒长度'),
    'HPVSWR': ('C_实验室数据', '6_感染_HPV_口腔'),
    'ORHPV': ('C_实验室数据', '6_感染_HPV_口腔检测'),
    'HPVP': ('C_实验室数据', '6_感染_HPV_私密部位'),
    'HPVSER': ('C_实验室数据', '6_感染_HPV_血清'),
    'HPVSWC': ('C_实验室数据', '6_感染_HPV_宫颈'),
    'HPVSRM': ('C_实验室数据', '6_感染_HPV_室温'),
    'MMRV': ('C_实验室数据', '6_感染_MMRV_疫苗'),
    
    # 7. 免疫反应
    'AL_IGE': ('C_实验室数据', '7_免疫_过敏原IgE'),
    'ALD': ('C_实验室数据', '7_免疫_过敏原'),
    'ALDS': ('C_实验室数据', '7_免疫_过敏原筛查'),
    'ALDUST': ('C_实验室数据', '7_免疫_过敏原尘螨'),
    'AGP': ('C_实验室数据', '7_免疫_酸性糖蛋白'),
    'SSAGP': ('C_实验室数据', '7_免疫_血清酸性糖蛋白'),
    'SS': ('C_实验室数据', '7_免疫_血清学'),
    'SSCMVG': ('C_实验室数据', '7_免疫_巨细胞病毒IgG'),
    'SSBFR': ('C_实验室数据', '7_免疫_血清抗体'),
    
    # 8. 环境暴露 - 重金属
    'PBCD': ('C_实验室数据', '8_环境暴露_重金属_铅镉'),
    'UHG': ('C_实验室数据', '8_环境暴露_重金属_尿汞'),
    'UHM': ('C_实验室数据', '8_环境暴露_重金属_尿汞甲基'),
    'IHGEM': ('C_实验室数据', '8_环境暴露_重金属_汞'),
    'CUSEZN': ('C_实验室数据', '8_环境暴露_重金属_铜硒锌'),
    
    # 8. 环境暴露 - 有机污染物
    'VOC': ('C_实验室数据', '8_环境暴露_有机物_挥发性'),
    'VOCWB': ('C_实验室数据', '8_环境暴露_有机物_全血挥发性'),
    'UVOC': ('C_实验室数据', '8_环境暴露_有机物_尿挥发性'),
    'DEET': ('C_实验室数据', '8_环境暴露_有机物_DEET'),
    'ETHOX': ('C_实验室数据', '8_环境暴露_有机物_乙氧基化合物'),
    'ETHOXS': ('C_实验室数据', '8_环境暴露_有机物_乙氧基代谢物'),
    'FLXCLN': ('C_实验室数据', '8_环境暴露_有机物_氟氯烃'),
    'FLDEP': ('C_实验室数据', '8_环境暴露_有机物_氟化物血浆'),
    'FLDEW': ('C_实验室数据', '8_环境暴露_有机物_氟化物水'),
    'PFC': ('C_实验室数据', '8_环境暴露_有机物_全氟化合物'),
    'PAH': ('C_实验室数据', '8_环境暴露_有机物_多环芳烃'),
    'PAHS': ('C_实验室数据', '8_环境暴露_有机物_多环芳烃代谢物'),
    'PHPYPA': ('C_实验室数据', '8_环境暴露_有机物_邻苯二甲酸盐酚类'),
    'PHTHTE': ('C_实验室数据', '8_环境暴露_有机物_邻苯二甲酸酯'),
    'FORMAS': ('C_实验室数据', '8_环境暴露_有机物_甲醛'),
    'FR': ('C_实验室数据', '8_环境暴露_有机物_阻燃剂'),
    
    # 8. 环境暴露 - 持久性有机污染物
    'PCBPOL': ('C_实验室数据', '8_环境暴露_持久性有机物_PCB'),
    'DOXPOL': ('C_实验室数据', '8_环境暴露_持久性有机物_二恶英'),
    'BFRPOL': ('C_实验室数据', '8_环境暴露_持久性有机物_溴化阻燃剂'),
    'POC': ('C_实验室数据', '8_环境暴露_持久性有机物'),
    
    # 8. 环境暴露 - 农药
    'PSTPOL': ('C_实验室数据', '8_环境暴露_农药_有机磷'),
    'UPHOPM': ('C_实验室数据', '8_环境暴露_农药_尿有机磷代谢物'),
    'PEST': ('C_实验室数据', '8_环境暴露_农药'),
    
    # 9. 毒物和药物
    'COT': ('C_实验室数据', '9_毒物_尼古丁代谢物'),
    'COTNAL': ('C_实验室数据', '9_毒物_尼古丁TSNAs'),
    'TSNA': ('C_实验室数据', '9_毒物_烟草特有亚硝胺'),
    'CRCO': ('C_实验室数据', '9_毒物_铬钴'),
    
    # 10. 特殊检测
    'FAS': ('C_实验室数据', '10_特殊检测_脂肪酸'),
    'AA': ('C_实验室数据', '10_特殊检测_芳香胺'),
    'AAS': ('C_实验室数据', '10_特殊检测_芳香胺代谢物'),
    'AMDGDS': ('C_实验室数据', '10_特殊检测_AMD基因分型'),
    'AMDGYD': ('C_实验室数据', '10_特殊检测_AMD基因型'),
    
    # 11. 基因和分子标志物
    'PUQMEC': ('C_实验室数据', '11_基因_杀虫剂问卷'),
    
    # 12. 其他特殊检测
    'WPIN': ('C_实验室数据', '12_特殊检测_水平蛋白'),
    'SEQ': ('C_实验室数据', '12_特殊检测_基因测序'),
    
    # L开头的实验室文件
    'L': ('C_实验室数据', '13_早期实验室数据'),
    'L06UAS': ('C_实验室数据', '13_早期实验室数据_尿砷'),
    'L13_2': ('C_实验室数据', '13_早期实验室数据_13_2'),
    'L28POC': ('C_实验室数据', '13_早期实验室数据_28POC'),
    'L36': ('C_实验室数据', '13_早期实验室数据_36'),
}

def parse_filename(filename):
    """
    解析NHANES文件名，提取前缀和后缀
    
    参数:
        filename: 文件名 (例如: 'DEMO_J.XPT', 'P_DEMO.XPT', 'WHQ.XPT')
    
    返回:
        tuple: (prefix, suffix) 例如: ('DEMO', '_J'), ('DEMO', 'P_'), ('WHQ', '')
    """
    # 移除文件扩展名并转为大写
    name = filename.upper().replace('.XPT', '')
    
    # 处理P_前缀的情况 (如 P_DEMO.XPT)
    if name.startswith('P_'):
        prefix = name[2:]  # 移除P_
        suffix = 'P_'
        return prefix, suffix
    
    # 处理标准后缀的情况 (如 DEMO_J.XPT)
    for suffix in ['_B', '_C', '_D', '_E', '_F', '_G', '_H', '_I', '_J', '_L']:
        if name.endswith(suffix):
            prefix = name[:-2]  # 移除后缀
            return prefix, suffix
    
    # 处理无后缀的情况 (如 WHQ.XPT)
    return name, ''

def find_category(prefix):
    """
    根据前缀查找文件类别
    
    参数:
        prefix: 文件前缀
        
    返回:
        tuple: (主类别, 子类别) 或 None
    """
    # 直接匹配
    if prefix in CATEGORY_MAP:
        return CATEGORY_MAP[prefix]
    
    # 模糊匹配 - 查找包含前缀的键
    for key in CATEGORY_MAP:
        if prefix.startswith(key) or key in prefix:
            return CATEGORY_MAP[key]
    
    # 特殊处理L开头的文件
    if prefix.startswith('L') and len(prefix) > 1:
        return CATEGORY_MAP['L']
    
    return None

def load_variable_info(table_name, variable_name):
    """
    加载变量的标签信息
    
    参数:
        table_name: 数据表名（如DEMO_J）
        variable_name: 变量名（如SEQN）
    
    返回:
        dict: 包含变量信息的字典，如果找不到则返回None
    """
    try:
        info_file = Path(VARIABLES_DIR) / table_name / variable_name / f"{variable_name}_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def load_variable_values(table_name, variable_name):
    """
    加载变量的数值标签映射
    
    参数:
        table_name: 数据表名（如DEMO_J）
        variable_name: 变量名（如RIAGENDR）
    
    返回:
        dict: 数值到标签的映射字典，如果找不到则返回None
    """
    try:
        values_file = Path(VARIABLES_DIR) / table_name / variable_name / f"{variable_name}_values.csv"
        if values_file.exists():
            df = pd.read_csv(values_file)
            # 创建从编码到描述的映射
            value_map = {}
            for _, row in df.iterrows():
                code = str(row['Code or Value']).strip()
                desc = str(row['Value Description']).strip()
                if code != '.' and code != 'nan':  # 排除缺失值
                    value_map[code] = desc
            return value_map
    except Exception:
        pass
    return None

def enhance_csv_with_labels(csv_file_path, table_name, logger):
    """
    增强CSV文件：替换列名为标签，替换数值为描述
    
    参数:
        csv_file_path: CSV文件路径
        table_name: 数据表名
        logger: 日志记录器
    
    返回:
        bool: 处理是否成功
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        original_columns = df.columns.tolist()
        
        # 1. 替换列名为标签
        column_rename_map = {}
        for col in df.columns:
            if col == 'YearRange':  # 跳过我们添加的年份列
                continue
                
            var_info = load_variable_info(table_name, col)
            if var_info and 'Label' in var_info:
                # 使用标签作为新列名，如果标签为空则保持原名
                label = var_info['Label'].strip()
                if label:
                    column_rename_map[col] = label
                    
        if column_rename_map:
            df = df.rename(columns=column_rename_map)
            logger.info(f"重命名列 {list(column_rename_map.keys())} -> {list(column_rename_map.values())}")
        
        # 2. 替换数值为描述
        enhanced_columns = 0
        for original_col in original_columns:
            if original_col == 'YearRange':  # 跳过年份列
                continue
                
            # 获取当前列名（可能已被重命名）
            current_col = column_rename_map.get(original_col, original_col)
            
            if current_col in df.columns:
                value_map = load_variable_values(table_name, original_col)
                if value_map:
                    # 将数值替换为描述
                    def replace_value(val):
                        if pd.isna(val):
                            return val
                        # 处理浮点数和整数
                        if isinstance(val, (int, float)):
                            # 尝试将浮点数转为整数字符串（如果是整数值）
                            if val == int(val):
                                val_str = str(int(val))
                            else:
                                val_str = str(val)
                        else:
                            val_str = str(val).strip()
                        return value_map.get(val_str, val)
                    
                    try:
                        original_values = df[current_col].dropna().unique()
                    except AttributeError:
                        # 如果列不存在，跳过
                        continue
                        
                    df[current_col] = df[current_col].apply(replace_value)
                    enhanced_columns += 1
                    
                    # 记录替换信息
                    replaced_values = []
                    for val in original_values:
                        if not pd.isna(val):
                            val_str = str(val).strip()
                            if val_str in value_map:
                                replaced_values.append(f"{val_str}->{value_map[val_str]}")
                    
                    if replaced_values:
                        logger.info(f"列 '{current_col}' 替换数值: {', '.join(replaced_values[:3])}" + 
                                  (f" 等{len(replaced_values)}项" if len(replaced_values) > 3 else ""))
        
        # 保存增强后的CSV文件
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"成功增强CSV文件: {csv_file_path.name} (重命名{len(column_rename_map)}列, 增强{enhanced_columns}列数值)")
        return True
        
    except Exception as e:
        logger.error(f"增强CSV文件 {csv_file_path.name} 时出错: {e}")
        return False

def convert_xpt_to_csv(xpt_file_path, csv_file_path, year_range, logger, enhance_labels=True):
    """
    将XPT文件转换为CSV格式，并标准化列名
    
    参数:
        xpt_file_path: XPT文件路径
        csv_file_path: 输出CSV文件路径
        year_range: 年份范围字符串
        logger: 日志记录器
        enhance_labels: 是否增强标签（替换列名和数值）
    
    返回:
        bool: 转换是否成功
    """
    try:
        # 读取XPT文件
        df = pd.read_sas(xpt_file_path, format='xport')
        
        # 标准化列名 - 只重命名存在的列
        columns_to_rename = {}
        for old_col, new_col in COLUMN_RENAME_MAP.items():
            if old_col in df.columns:
                columns_to_rename[old_col] = new_col
        
        if columns_to_rename:
            df = df.rename(columns=columns_to_rename)
            logger.info(f"标准化列名: {list(columns_to_rename.keys())}")
        
        # 添加年份范围列
        df['YearRange'] = year_range
        
        # 创建输出目录
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV，使用UTF-8编码
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"转换成功: {xpt_file_path.name} -> {csv_file_path}")
        
        # 如果启用标签增强，则应用变量标签和数值标签
        if enhance_labels:
            # 从文件名提取表名（去掉后缀）
            filename = xpt_file_path.stem.upper()
            # 解析表名（去掉年份后缀）
            prefix, suffix = parse_filename(filename + '.XPT')
            
            # 构建正确的表名
            if suffix == 'P_':
                table_name = 'P_' + prefix
            else:
                table_name = prefix + suffix
            
            logger.info(f"尝试增强文件 {csv_file_path.name}，表名: {table_name}")
            
            # 应用标签增强
            enhance_csv_with_labels(csv_file_path, table_name, logger)
        
        return True
        
    except Exception as e:
        logger.error(f"转换文件 {xpt_file_path.name} 时出错: {e}")
        return False

def merge_category_files(category_dir, logger):
    """
    合并类别目录下的所有CSV文件为单个文件
    
    参数:
        category_dir: 类别目录路径
        logger: 日志记录器
    """
    category_path = Path(category_dir)
    csv_files = list(category_path.glob("*.csv"))
    
    if not csv_files:
        return
    
    logger.info(f"处理类别: {category_path.name}")
    
    merged_dataframes = []
    
    for csv_file in csv_files:
        try:
            # 跳过已合并的文件
            if '_merged' in csv_file.name:
                continue
                
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 从文件名提取年份范围（如果没有YearRange列）
            if 'YearRange' not in df.columns:
                # 文件名格式: PREFIX_YEAR-RANGE.csv
                filename_parts = csv_file.stem.split('_')
                if len(filename_parts) >= 2:
                    year_range = filename_parts[-1]
                    df['YearRange'] = year_range
            
            merged_dataframes.append(df)
            
        except Exception as e:
            logger.error(f"读取文件 {csv_file.name} 时出错: {e}")
    
    if merged_dataframes:
        # 合并所有DataFrame
        merged_df = pd.concat(merged_dataframes, ignore_index=True, sort=False)
        
        # 生成合并文件名 - 使用目录名（已包含前缀）
        merged_filename = f"{category_path.name}_merged.csv"
        merged_filepath = category_path / merged_filename
        
        # 保存合并文件
        merged_df.to_csv(merged_filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"合并完成: {len(csv_files)} 个文件 -> {merged_filename}, 总记录数: {len(merged_df)}")
    else:
        logger.warning(f"没有找到可合并的CSV文件")

def process_xpt_to_csv_raw(source_root_dir, output_dir, logger):
    """
    处理所有XPT文件，转换为CSV格式（不含标签增强）并组织到新的目录结构
    
    参数:
        source_root_dir: 源根目录
        output_dir: 输出目录
        logger: 日志记录器
    """
    source_path = Path(source_root_dir)
    output_path = Path(output_dir)
    
    logger.info("开始XPT到CSV转换（原始格式，不含标签增强）...")
    
    # 统计信息
    stats = {
        'processed': 0,
        'success': 0,
        'errors': 0,
        'skipped': 0
    }
    
    # 递归查找所有XPT文件
    xpt_files = list(source_path.rglob("*.xpt")) + list(source_path.rglob("*.XPT"))
    
    logger.info(f"找到 {len(xpt_files)} 个XPT文件")
    
    for xpt_file in tqdm(xpt_files, desc="转换XPT文件（原始格式）"):
        try:
            filename = xpt_file.name
            stats['processed'] += 1
            
            # 解析文件名
            prefix, suffix = parse_filename(filename)
            year_range = YEAR_MAP.get(suffix, '未知年份')
            
            # 查找类别
            category_info = find_category(prefix)
            
            if not category_info:
                logger.warning(f"跳过未识别文件: {filename}")
                stats['skipped'] += 1
                continue
            
            main_category, sub_category = category_info
            
            # 构建CSV文件路径 - 使用前缀_子类别作为目录名
            csv_filename = f"{prefix}_{year_range}.csv"
            csv_output_dir = output_path / main_category / f"{prefix}_{sub_category}"
            csv_file_path = csv_output_dir / csv_filename
            
            # 转换文件（禁用标签增强）
            if convert_xpt_to_csv(xpt_file, csv_file_path, year_range, logger, enhance_labels=False):
                stats['success'] += 1
            else:
                stats['errors'] += 1
                
        except Exception as e:
            stats['errors'] += 1
            logger.error(f"处理文件 {filename} 时出错: {e}")
    
    # 输出统计信息
    logger.info("="*80)
    logger.info("XPT转CSV统计（原始格式）:")
    logger.info(f"处理总数: {stats['processed']}")
    logger.info(f"转换成功: {stats['success']}")
    logger.info(f"转换失败: {stats['errors']}")
    logger.info(f"跳过文件: {stats['skipped']}")
    
    return stats

def process_xpt_to_csv(source_root_dir, output_dir, logger):
    """
    处理所有XPT文件，转换为CSV格式并组织到新的目录结构
    
    参数:
        source_root_dir: 源根目录
        output_dir: 输出目录
        logger: 日志记录器
    """
    source_path = Path(source_root_dir)
    output_path = Path(output_dir)
    
    logger.info("开始XPT到CSV转换...")
    
    # 统计信息
    stats = {
        'processed': 0,
        'success': 0,
        'errors': 0,
        'skipped': 0
    }
    
    # 递归查找所有XPT文件
    xpt_files = list(source_path.rglob("*.xpt")) + list(source_path.rglob("*.XPT"))
    
    logger.info(f"找到 {len(xpt_files)} 个XPT文件")
    
    for xpt_file in tqdm(xpt_files, desc="转换XPT文件"):
        try:
            filename = xpt_file.name
            stats['processed'] += 1
            
            # 解析文件名
            prefix, suffix = parse_filename(filename)
            year_range = YEAR_MAP.get(suffix, '未知年份')
            
            # 查找类别
            category_info = find_category(prefix)
            
            if not category_info:
                logger.warning(f"跳过未识别文件: {filename}")
                stats['skipped'] += 1
                continue
            
            main_category, sub_category = category_info
            
            # 构建CSV文件路径 - 使用前缀_子类别作为目录名
            csv_filename = f"{prefix}_{year_range}.csv"
            csv_output_dir = output_path / main_category / f"{prefix}_{sub_category}"
            csv_file_path = csv_output_dir / csv_filename
            
            # 转换文件（默认启用标签增强）
            if convert_xpt_to_csv(xpt_file, csv_file_path, year_range, logger, enhance_labels=True):
                stats['success'] += 1
            else:
                stats['errors'] += 1
                
        except Exception as e:
            stats['errors'] += 1
            logger.error(f"处理文件 {filename} 时出错: {e}")
    
    # 输出统计信息
    logger.info("="*80)
    logger.info("XPT转CSV统计:")
    logger.info(f"处理总数: {stats['processed']}")
    logger.info(f"转换成功: {stats['success']}")
    logger.info(f"转换失败: {stats['errors']}")
    logger.info(f"跳过文件: {stats['skipped']}")
    
    return stats

def merge_all_categories(csv_root_dir, logger):
    """
    合并所有类别的年度数据
    
    参数:
        csv_root_dir: CSV根目录
        logger: 日志记录器
    """
    csv_path = Path(csv_root_dir)
    
    if not csv_path.exists():
        logger.error(f"CSV目录不存在: {csv_root_dir}")
        return
    
    logger.info("开始合并年度数据...")
    
    # 查找所有最底层的子目录（包含CSV文件的目录）
    category_dirs = []
    for root, dirs, files in os.walk(csv_path):
        # 如果目录包含CSV文件且没有子目录，则认为是类别目录
        csv_files = [f for f in files if f.endswith('.csv') and not f.endswith('_merged.csv')]
        if csv_files and not dirs:
            category_dirs.append(Path(root))
    
    logger.info(f"找到 {len(category_dirs)} 个类别目录")
    
    for category_dir in tqdm(category_dirs, desc="合并类别数据"):
        merge_category_files(category_dir, logger)
    
    # 创建总合并目录
    create_master_merged_directory(csv_root_dir, logger)
    
    logger.info("所有类别数据合并完成！")

def create_master_merged_directory(csv_root_dir, logger):
    """
    创建总的合并文件目录，收集所有按前缀分组的合并文件
    
    参数:
        csv_root_dir: CSV根目录
        logger: 日志记录器
    """
    csv_path = Path(csv_root_dir)
    
    # 创建输出目录（修改命名以区分新旧格式）
    output_dir_name = f"{csv_path.name}_merged_by_prefix"
    output_dir = csv_path.parent / output_dir_name
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"创建总合并目录: {output_dir_name}")
    
    # 查找所有合并文件
    merged_files = list(csv_path.rglob("*_merged.csv"))
    
    copied_count = 0
    for merged_file in merged_files:
        try:
            # 复制文件到总目录
            dest_file = output_dir / merged_file.name
            shutil.copy2(merged_file, dest_file)
            copied_count += 1
            
        except Exception as e:
            logger.error(f"复制合并文件失败 {merged_file.name}: {e}")
    
    logger.info(f"总合并目录创建完成，复制了 {copied_count} 个合并文件到 {output_dir_name}")
    
    return output_dir

def enhance_existing_csvs(csv_root_dir, logger):
    """
    对现有的CSV文件应用标签增强
    
    参数:
        csv_root_dir: CSV根目录
        logger: 日志记录器
    """
    csv_path = Path(csv_root_dir)
    
    if not csv_path.exists():
        logger.error(f"CSV目录不存在: {csv_root_dir}")
        return
    
    logger.info("开始增强现有CSV文件的标签...")
    
    # 统计信息
    stats = {
        'processed': 0,
        'success': 0,
        'errors': 0,
        'skipped': 0
    }
    
    # 递归查找所有CSV文件
    csv_files = list(csv_path.rglob("*.csv"))
    
    # 过滤掉合并文件
    csv_files = [f for f in csv_files if '_merged' not in f.name]
    
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in tqdm(csv_files, desc="增强CSV文件"):
        try:
            stats['processed'] += 1
            
            # 从文件名推断表名
            filename = csv_file.stem  # 例如: DEMO_2017-2018
            
            # 移除年份后缀，获取表名
            parts = filename.split('_')
            if len(parts) >= 2:
                table_name = '_'.join(parts[:-1])  # 去掉最后的年份部分
            else:
                table_name = parts[0]
            
            # 检查是否存在对应的变量描述文件夹
            table_dir = Path(VARIABLES_DIR) / table_name
            if not table_dir.exists():
                logger.warning(f"跳过文件（未找到变量描述）: {csv_file.name}")
                stats['skipped'] += 1
                continue
            
            # 应用标签增强
            if enhance_csv_with_labels(csv_file, table_name, logger):
                stats['success'] += 1
            else:
                stats['errors'] += 1
                
        except Exception as e:
            stats['errors'] += 1
            logger.error(f"处理CSV文件 {csv_file.name} 时出错: {e}")
    
    # 输出统计信息
    logger.info("="*80)
    logger.info("CSV标签增强统计:")
    logger.info(f"处理总数: {stats['processed']}")
    logger.info(f"增强成功: {stats['success']}")
    logger.info(f"增强失败: {stats['errors']}")
    logger.info(f"跳过文件: {stats['skipped']}")
    
    return stats

def auto_process_nhanes_data():
    """
    自动处理NHANES数据的主函数
    """
    # 设置日志
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("NHANES数据自动处理开始")
    logger.info(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # 检查输入目录
    if not os.path.exists(DEST_DIR):
        logger.error(f"数据目录不存在: {DEST_DIR}")
        return False
    
    # 检查变量描述目录
    if not os.path.exists(VARIABLES_DIR):
        logger.warning(f"变量描述目录不存在: {VARIABLES_DIR}")
        logger.warning("将跳过标签增强功能")
    
    try:
        # 步骤1: 转换XPT到CSV
        logger.info("步骤1: 开始XPT到CSV转换...")
        stats = process_xpt_to_csv(DEST_DIR, CSV_OUTPUT_DIR, logger)
        
        if stats['success'] == 0:
            logger.warning("没有成功转换任何文件，跳过后续步骤")
            return False
        
        # 步骤2: 合并年度数据
        logger.info("步骤2: 开始合并年度数据...")
        merge_all_categories(CSV_OUTPUT_DIR, logger)
        
        logger.info("="*60)
        logger.info("NHANES数据处理完成！")
        logger.info(f"输出目录: {CSV_OUTPUT_DIR}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return False

def organize_files(source_dir, dest_dir, dry_run=False):
    """
    整理NHANES文件
    
    参数:
        source_dir: 源目录路径
        dest_dir: 目标目录路径
        dry_run: 是否只是模拟运行，不实际移动文件
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"❌ 源目录不存在: {source_dir}")
        return
    
    # 获取所有.xpt文件
    xpt_files = list(source_path.glob("*.xpt")) + list(source_path.glob("*.XPT"))
    
    if not xpt_files:
        print(f"❌ 在源目录中未找到.xpt文件: {source_dir}")
        return
    
    print(f"📁 源目录: {source_dir}")
    print(f"📁 目标目录: {dest_dir}")
    print(f"📊 找到 {len(xpt_files)} 个.xpt文件")
    
    if dry_run:
        print("🔍 模拟运行模式 - 不会实际移动文件")
    
    print("\n" + "="*80)
    
    # 统计信息
    stats = {
        'moved': 0,
        'uncategorized': 0,
        'errors': 0
    }
    
    # 处理每个文件
    for file_path in tqdm(xpt_files, desc="处理文件"):
        try:
            filename = file_path.name
            
            # 解析文件名
            prefix, suffix = parse_filename(filename)
            
            # 查找年份
            year = YEAR_MAP.get(suffix, '未知年份')
            
            # 查找类别
            category_info = find_category(prefix)
            
            if category_info:
                main_category, sub_category = category_info
                
                # 构建目标路径
                target_dir = dest_path / main_category / sub_category / year
                target_file = target_dir / filename
                
                # 创建目录
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                
                # 移动文件
                if not dry_run:
                    if target_file.exists():
                        print(f"⚠️  目标文件已存在，跳过: {filename}")
                        continue
                    shutil.move(str(file_path), str(target_file))
                
                stats['moved'] += 1
                print(f"✅ 已移动: {filename} -> {main_category}/{sub_category}/{year}/")
                
            else:
                # 无法识别的文件放到_Uncategorized目录
                uncategorized_dir = dest_path / "_Uncategorized"
                target_file = uncategorized_dir / filename
                
                if not dry_run:
                    uncategorized_dir.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        shutil.move(str(file_path), str(target_file))
                
                stats['uncategorized'] += 1
                print(f"⚠️  未识别文件: {filename} -> _Uncategorized/")
                
        except Exception as e:
            stats['errors'] += 1
            print(f"❌ 处理文件 {filename} 时出错: {e}")
    
    # 输出统计信息
    print("\n" + "="*80)
    print("📊 处理统计:")
    print(f"✅ 成功移动: {stats['moved']} 个文件")
    print(f"⚠️  未识别文件: {stats['uncategorized']} 个文件")
    print(f"❌ 错误: {stats['errors']} 个文件")
    print(f"📁 总计: {len(xpt_files)} 个文件")
    
    if not dry_run:
        print(f"\n🎉 文件整理完成！请查看目标目录: {dest_dir}")
    else:
        print(f"\n🔍 模拟运行完成！如需实际移动文件，请设置 dry_run=False")

def main():
    """主函数"""
    print("🏥 NHANES 数据处理和预处理工具")
    print("="*60)
    
    print("功能选项:")
    print("1. 整理XPT文件到分类目录")
    print("2. 将XPT文件转换为CSV格式（含标签增强，按前缀分文件夹）") 
    print("3. 将XPT文件转换为CSV格式（不含标签增强，按前缀分文件夹）")
    print("4. 合并类别年度数据（按前缀分别合并）")
    print("5. 完整处理流程 (XPT -> CSV -> 合并)")
    print("6. 仅处理现有已组织的数据 (XPT -> CSV -> 合并)")
    print("7. 增强现有CSV文件的标签和数值描述")
    print("")
    print("注意：新版本会将每个变量前缀放在独立的文件夹中")
    print("     例如：DR1IFF_2_膳食数据_单个食物/，DR2IFF_2_膳食数据_单个食物/")
    
    choice = input("\n请选择功能 (1-7): ").strip()
    
    if choice == "1":
        # 原有的文件整理功能
        if not os.path.exists(SOURCE_DIR):
            print(f"❌ 源目录不存在: {SOURCE_DIR}")
            print("请检查配置区域中的 SOURCE_DIR 设置")
            return
        
        print(f"源目录: {SOURCE_DIR}")
        print(f"目标目录: {DEST_DIR}")
        
        response = input("\n是否先进行模拟运行查看效果？(y/n): ").lower().strip()
        
        if response == 'y':
            print("\n🔍 开始模拟运行...")
            organize_files(SOURCE_DIR, DEST_DIR, dry_run=True)
            
            response2 = input("\n模拟运行完成。是否继续实际移动文件？(y/n): ").lower().strip()
            if response2 == 'y':
                print("\n📦 开始实际移动文件...")
                organize_files(SOURCE_DIR, DEST_DIR, dry_run=False)
            else:
                print("操作取消。")
        else:
            print("\n📦 开始移动文件...")
            organize_files(SOURCE_DIR, DEST_DIR, dry_run=False)
    
    elif choice == "2":
        # XPT到CSV转换（含标签增强）
        source_input = input(f"\n请输入XPT文件源目录 (默认: {DEST_DIR}): ").strip()
        if not source_input:
            source_input = DEST_DIR
            
        if not os.path.exists(source_input):
            print(f"❌ 源目录不存在: {source_input}")
            return
            
        print(f"源目录: {source_input}")
        print(f"CSV输出目录: {CSV_OUTPUT_DIR}")
        
        confirm = input("\n确认开始转换？(y/n): ").lower().strip()
        if confirm == 'y':
            logger = setup_logging()
            process_xpt_to_csv(source_input, CSV_OUTPUT_DIR, logger)
        else:
            print("操作取消。")
    
    elif choice == "3":
        # XPT到CSV转换（不含标签增强，原始格式）
        source_input = input(f"\n请输入XPT文件源目录 (默认: {DEST_DIR}): ").strip()
        if not source_input:
            source_input = DEST_DIR
            
        if not os.path.exists(source_input):
            print(f"❌ 源目录不存在: {source_input}")
            return
            
        print(f"源目录: {source_input}")
        print(f"CSV输出目录: {CSV_OUTPUT_DIR_RAW}")
        print("注意：此功能不会进行标签增强，保持原始变量名和数值编码")
        
        confirm = input("\n确认开始转换？(y/n): ").lower().strip()
        if confirm == 'y':
            logger = setup_logging()
            # 调用process_xpt_to_csv_raw函数，不启用标签增强
            process_xpt_to_csv_raw(source_input, CSV_OUTPUT_DIR_RAW, logger)
        else:
            print("操作取消。")
    
    elif choice == "4":
        # 合并类别数据
        csv_input = input(f"\n请输入CSV目录 (默认: {CSV_OUTPUT_DIR}): ").strip()
        if not csv_input:
            csv_input = CSV_OUTPUT_DIR
            
        if not os.path.exists(csv_input):
            print(f"❌ CSV目录不存在: {csv_input}")
            return
            
        print(f"CSV目录: {csv_input}")
        
        confirm = input("\n确认开始合并？(y/n): ").lower().strip()
        if confirm == 'y':
            logger = setup_logging()
            merge_all_categories(csv_input, logger)
        else:
            print("操作取消。")
    
    elif choice == "5":
        # 完整处理流程
        if not os.path.exists(SOURCE_DIR):
            print(f"❌ 源目录不存在: {SOURCE_DIR}")
            print("请检查配置区域中的 SOURCE_DIR 设置")
            return
            
        print("完整处理流程:")
        print(f"1. 整理文件: {SOURCE_DIR} -> {DEST_DIR}")
        print(f"2. 转换格式: {DEST_DIR} -> {CSV_OUTPUT_DIR}")
        print(f"3. 合并数据: {CSV_OUTPUT_DIR}")
        
        confirm = input("\n确认开始完整处理？(y/n): ").lower().strip()
        if confirm == 'y':
            logger = setup_logging()
            
            print("\n步骤1: 整理XPT文件...")
            organize_files(SOURCE_DIR, DEST_DIR, dry_run=False)
            
            print("\n步骤2: 转换为CSV格式...")
            process_xpt_to_csv(DEST_DIR, CSV_OUTPUT_DIR, logger)
            
            print("\n步骤3: 合并年度数据...")
            merge_all_categories(CSV_OUTPUT_DIR, logger)
            
            print("\n🎉 完整处理流程完成！")
        else:
            print("操作取消。")
    
    elif choice == "6":
        # 处理现有已组织的数据
        source_input = input(f"\n请输入已组织的NHANES数据目录 (默认: {DEST_DIR}): ").strip()
        if not source_input:
            source_input = DEST_DIR
            
        if not os.path.exists(source_input):
            print(f"❌ 目录不存在: {source_input}")
            return
            
        print("处理现有数据:")
        print(f"1. 转换格式: {source_input} -> {CSV_OUTPUT_DIR}")
        print(f"2. 合并数据: {CSV_OUTPUT_DIR}")
        
        confirm = input("\n确认开始处理？(y/n): ").lower().strip()
        if confirm == 'y':
            logger = setup_logging()
            
            print("\n步骤1: 转换为CSV格式...")
            process_xpt_to_csv(source_input, CSV_OUTPUT_DIR, logger)
            
            print("\n步骤2: 合并年度数据...")
            merge_all_categories(CSV_OUTPUT_DIR, logger)
            
            print("\n🎉 处理完成！")
        else:
            print("操作取消。")
    
    elif choice == "7":
        # 增强现有CSV文件标签
        csv_input = input(f"\n请输入CSV目录 (默认: {CSV_OUTPUT_DIR}): ").strip()
        if not csv_input:
            csv_input = CSV_OUTPUT_DIR
            
        if not os.path.exists(csv_input):
            print(f"❌ CSV目录不存在: {csv_input}")
            return
            
        # 检查变量描述目录
        if not os.path.exists(VARIABLES_DIR):
            print(f"❌ 变量描述目录不存在: {VARIABLES_DIR}")
            print("请确保NHANES_Variables文件夹存在且包含变量描述信息")
            return
            
        print(f"CSV目录: {csv_input}")
        print(f"变量描述目录: {VARIABLES_DIR}")
        print("\n此功能将：")
        print("- 将变量名替换为更易读的标签")
        print("- 将数值编码替换为对应的文本描述")
        
        confirm = input("\n确认开始增强？(y/n): ").lower().strip()
        if confirm == 'y':
            logger = setup_logging()
            enhance_existing_csvs(csv_input, logger)
            print("\n🎉 标签增强完成！")
        else:
            print("操作取消。")
    
    else:
        print("❌ 无效选择，请重新运行程序。")

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # 交互式模式
        main()
    else:
        # 默认自动处理模式
        print("🏥 NHANES数据自动处理模式")
        print("如需交互式选择，请使用: python organize_nhanes_files.py --interactive")
        print("="*60)
        
        success = auto_process_nhanes_data()
        
        if success:
            print("\n✅ 处理完成！请查看日志文件获取详细信息。")
        else:
            print("\n❌ 处理失败！请查看日志文件获取错误信息。")
            sys.exit(1) 