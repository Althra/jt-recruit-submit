import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm

# 定义数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将CIFAR-10的32x32图像调整到224x224以适应ViT的输入要求
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载训练和测试数据集
train_dataset = datasets.CIFAR10(root='./datasets/cifar10', train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root='./datasets/cifar10', train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"Dateset size: {len(train_dataset), len(test_dataset)}")

# 加载预训练的ViT模型
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)

# 将模型转移到GPU（如果可用）
device = torch.device("cuda")
model = model.to(device)
print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        test_model(model, test_loader, device)
        model.train()

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    import time
    start_time = time.time()
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    end_time = time.time()
    print(f"Training time for ViT: {end_time - start_time:.2f} seconds")
    torch.save(model.state_dict(), 'vit_cifar10.pth')
