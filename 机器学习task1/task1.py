import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 数据预处理
data = pd.read_csv('~/datasets/boston-housing/train.csv')
X = data.drop(['medv', 'ID'], axis=1).values
y = data['medv'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
train_loader = DataLoader((X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# 定义模型
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
model = HousePriceModel(input_dim)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test MSE Loss: {test_loss.item():.4f}')


# 使用numpy分区间评估
import numpy as np

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()
    true_values = y_test_tensor.numpy().flatten()

# 将真实值和预测值组合到一个DataFrame中
results = pd.DataFrame({
    'True': true_values,
    'Predicted': predictions
})

# 设定价格区间
bins = [0, 20, 35, np.max(true_values)]
labels = ['Low', 'Medium', 'High']
results['Price_Range'] = pd.cut(results['True'], bins=bins, labels=labels)

# 计算每个价格区间的均方误差
mse_per_range = results.groupby('Price_Range').apply(
    lambda x: np.mean((x['True'] - x['Predicted'])**2)
)
print(f"MSE per Price Range:\n {mse_per_range}")

