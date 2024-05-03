# python设计/home/liuweilong/MMSA-FET/MOSI/label.csv转换为xls
import pandas as pd

# 读取CSV文件
data = pd.read_csv('/home/liuweilong/MMSA-FET/MOSI/label.csv')

# 将数据保存为XLS文件
data.to_excel('/home/liuweilong/MMSA-FET/MOSI/label.xlsx', index=False)