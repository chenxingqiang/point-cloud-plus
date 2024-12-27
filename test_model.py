import torch
from pointtransform import PointTransformerCls, arg

def test_model():
    # 创建一个随机的点云数据
    # batch_size = 2, num_points = 1024, features = 6 (xyz + 特征)
    point_cloud = torch.rand(2, 1024, 6)
    
    # 创建模型配置
    cfg = arg()
    
    # 初始化模型
    model = PointTransformerCls(cfg)
    
    # 前向传播
    output = model(point_cloud)
    
    print("Input shape:", point_cloud.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

if __name__ == "__main__":
    test_model() 