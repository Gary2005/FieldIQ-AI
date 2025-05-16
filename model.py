import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)  # 输出 (32, 270, 480)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # 输出 (64, 135, 240)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 输出 (128, 68, 120)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 输出 (256, 34, 60)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 输出 (512, 17, 30)
        self.bn5 = nn.BatchNorm2d(512)
        
        # 自适应平均池化，将特征图降为 (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 最后全连接层输出9维向量
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # 示例测试：创建随机输入，查看输出形状
    model = HomographyNet()
    input_tensor = torch.randn(1, 3, 540, 960)
    output = model(input_tensor)
    print("Output shape:", output)  # 应输出 [1, 9]

class HomographyModel_res_net(nn.Module):
    def __init__(self):
        super(HomographyModel_res_net, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )

    def forward(self, x):
        features = self.cnn(x).flatten(1)
        output = self.fc(features)
        output = nn.functional.normalize(output, p=2, dim=1)
        return output.view(-1, 9)

    def cosine_loss(self, pred, target):
        return 1 - torch.mean(torch.sum(pred * target, dim = -1))