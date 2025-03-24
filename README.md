# 城市探险家：街景字符识别赛

## 数据集下载
运行 download.py 下载数据集到tcdata文件夹

## 单模型分数

1. 以mobilenet_v2 为预训练模型: 0.8795

2. ResNet50: 0.8937

3. ResNet101: 0.9012

   ![image-20250323003411807](https://img-md-js.linjsblog.top/img/202503230034893.png)

## 融合模型分数

ResNet101+MobileNet_v2: 0.9105

![image-20250325003206897](https://img-md-js.linjsblog.top/img/202503250125982.png)