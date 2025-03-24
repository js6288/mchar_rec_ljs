from torch import nn
from torchvision.models import resnet18,resnet50,ResNet18_Weights, ResNet50_Weights
from torchvision.models.mobilenet import mobilenet_v2,MobileNet_V2_Weights
from Config import Config

config = Config()
# 自定义网络
# 以ResNet50 为主干网络
class DigitsResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        #resnet50
        #self.net = resnet50(pretrained=True)  # deprecated
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 移除原始ResNet50的最后全连接层，保留特征提取层
        # children()[:-1] 表示取除最后一层外的所有层
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        self.cnn = self.net

        # 定义4个隐藏全连接层（每个对应一个字符位置）
        self.hd_fc1 = nn.Linear(2048, 128)
        self.hd_fc2 = nn.Linear(2048, 128)
        self.hd_fc3 = nn.Linear(2048, 128)
        self.hd_fc4 = nn.Linear(2048, 128)

        # Dropout层防止过拟合
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.25)
        # 最终分类层（每个位置输出11类：0-9数字 + 空白符）
        self.fc1 = nn.Linear(128, config.class_num)
        self.fc2 = nn.Linear(128, config.class_num)
        self.fc3 = nn.Linear(128, config.class_num)
        self.fc4 = nn.Linear(128, config.class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1) #展平

        # 多任务分支（4个独立分类器）
        feat1 = self.dropout_1(self.hd_fc1(feat)) #降维+dropout
        feat2 = self.dropout_2(self.hd_fc2(feat))
        feat3 = self.dropout_3(self.hd_fc3(feat))
        feat4 = self.dropout_4(self.hd_fc4(feat))

        # 分类输出（4个位置的预测结果）
        c1 = self.fc1(feat1)  # 每个输出形状: (batch, 11)
        c2 = self.fc2(feat2)
        c3 = self.fc3(feat3)
        c4 = self.fc4(feat4)

        return c1, c2, c3, c4


class DigitsResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        #resNet18
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 移除原始全连接层（fc），替换为Identity层（即无操作）
        self.net.fc = nn.Identity()
        # 定义BatchNorm层
        self.bn = nn.BatchNorm1d(512)

        # 四个独立连接层
        self.fc1 = nn.Linear(512, config.class_num)
        self.fc2 = nn.Linear(512, config.class_num)
        self.fc3 = nn.Linear(512, config.class_num)
        self.fc4 = nn.Linear(512, config.class_num)

    def forward(self, img):
        # feature = self.net(img).squeeze()
        features = self.net(img)
        features = features.view(features.shape[0], -1)
        features = self.bn(features) #添加BatchNorm

        fc1 = self.fc1(features)
        fc2 = self.fc2(features)
        fc3 = self.fc3(features)
        fc4 = self.fc4(features)

        return fc1, fc2, fc3, fc4

# modelNet_V2为主干网络
class DigitsMobileNet(nn.Module):
    def __init__(self,class_num=11):
        super().__init__()
        # 加载预训练MobileNetV2的特征层（去掉分类层）
        self.net = mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1).features
        # 自适应池化层（输出形状：1x1）
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 批量归一化
        self.bn = nn.BatchNorm1d(1280)

        # 四个独立全连接层（每个对应一个字符位置）
        self.fc1 = nn.Linear(1280, class_num)
        self.fc2 = nn.Linear(1280, class_num)
        self.fc3 = nn.Linear(1280, class_num)
        self.fc4 = nn.Linear(1280, class_num)
    def forward(self, img):
        # 特征提取与池化
        features = self.avgpool(self.net(img))  # 输出形状：(batch, 1280, 1, 1)
        features = features.view(-1, 1280)      # 展平为 (batch, 1280)
        features = self.bn(features)            # 批归一化

        # 四个分支的分类结果
        fc1 = self.fc1(features)
        fc2 = self.fc2(features)
        fc3 = self.fc3(features)
        fc4 = self.fc4(features)

        return fc1, fc2, fc3, fc4



