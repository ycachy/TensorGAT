# check_fixed.py
import torch
import sys
import os

# 确保 PairData 类在 torch.load 之前被定义
from torch_geometric.data import Data

class PairData(Data):
    def __inc__(self, key, value):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0

# 临时将当前模块注册为 __main__
# 这样 torch.load 就能找到 PairData 类了
sys.modules['__main__'].PairData = PairData

# 现在加载数据
data = torch.load(r"J:\code\processed\1\data_0.pt")

print("数据加载成功！")
print(f"数据包含的属性: {dir(data)}")
print(f"x1 shape: {data.x1.shape if hasattr(data, 'x1') else 'N/A'}")
print(f"x2 shape: {data.x2.shape if hasattr(data, 'x2') else 'N/A'}")
print(f"edge_index1 shape: {data.edge_index1.shape if hasattr(data, 'edge_index1') else 'N/A'}")
print(f"edge_index2 shape: {data.edge_index2.shape if hasattr(data, 'edge_index2') else 'N/A'}")
print(f"y: {data.y if hasattr(data, 'y') else 'N/A'}")