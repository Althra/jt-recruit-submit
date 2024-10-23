# Task 1: 波士顿房价预测

## 1. 数据预处理

- 特征选择：数据集中除`ID`, `MEDV`外的所有特征作为输入，`MEDV` 作为输出。
- 数据分割:
  - 训练集：80%
  - 测试集：20%
- 标准化：对所有特征进行标准化处理，以提高模型的收敛速度。
- 转化为张量：将数据转化为训练用的张量。

```python
data = pd.read_csv('~/datasets/boston-housing/train.csv')
X = data.drop(['medv', 'ID'], axis=1).values
y = data['medv'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor ...
```

## 2. 模型结构

- 模型类型：多层感知机（MLP）
- 输入层：13 个特征
- 隐藏层:
  - 第一层：64 个神经元，激活函数为 ReLU
  - 第二层：32 个神经元，激活函数为 ReLU
- 输出层：1 个神经元，用于回归预测房价

```python
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
```

## 3. 训练过程

- 损失函数：均方误差（MSE）
- 优化器：Adam，学习率 `0.001`
- 批次大小：32
- 训练轮数：100 轮

```python
# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
# 训练过程
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

训练结果：

| Epoch | Loss     |
| ----- | -------- |
| 10    | 137.5570 |
| 20    | 29.6474  |
| 50    | 18.1681  |
| 100   | 9.3372   |

## 4. 结果评估

### 1. 整体评估

- 测试集 MSE：11.2877

```python
# 测试集评估
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test MSE Loss：{test_loss.item():.4f}')
```

### 2. 不同价格区间的评估

- 价格区间划分：低价 (0-20)、中价 (20-35)、高价 (>35)
- 每个区间的 MSE:

| Price Range | MSE       |
| ----------- | --------- |
| Low         | 8.563674  |
| Medium      | 7.398247  |
| High        | 45.391586 |

```python
# 计算每个价格区间的均方误差（MSE）
mse_per_range = results.groupby('Price_Range').apply(
    lambda x：np.mean((x['True'] - x['Predicted'])2)
)
```

