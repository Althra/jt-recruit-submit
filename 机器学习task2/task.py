import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size = 128 # 一次前向或反向传播所处理的数据样本的数量
learning_rate = 0.001 # 学习率
num_epochs = 15 # 训练轮数

transform = transforms.Compose([
    transforms.ToTensor()
])

# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='~/datasets/cifar10', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='~/datasets/cifar10', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # 卷积层1
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1) # 卷积层2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 卷积层3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 池化层
        self.fc1 = nn.Linear(128 * 4 * 4, 512) # 全连接层1
        self.fc2 = nn.Linear(512, 10) # 全连接层2

    def forward(self, x):
        # TODO:这里补全你的前向传播
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # 展平
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# TODO:补全
model = Network().to(device)

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器

def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()
