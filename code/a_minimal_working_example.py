# a_minimal_working_example.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

# --- 配置 ---
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10 # 只跑10轮，快速验证

# --- 数据集 ---
class SimpleBMIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        image = Image.open(img_path).convert('RGB')
        bmi = torch.tensor(float(self.annotations.iloc[index, 1]), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, bmi

transform = transforms.Compose([
    transforms.Resize((128, 128)), # 用更小的尺寸，加快速度
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = SimpleBMIDataset(TRAIN_CSV, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 最简模型 ---
class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.flat = nn.Flatten()
        # 128 -> 64 -> 32
        self.fc1 = nn.Linear(32 * 32 * 32, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flat(x)
        x = self.fc1(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToyCNN().to(device)

# --- 训练组件 ---
criterion = nn.MSELoss() # 回归最基础的 MSE
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 极简训练循环 ---
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # 前向
        scores = model(data).squeeze(-1)
        loss = criterion(scores, targets)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        
        # 更新
        optimizer.step()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

print("\nMinimal training finished.")

model.eval()
with torch.no_grad():
    # 取一个batch的验证数据
    val_dataset = SimpleBMIDataset(VAL_CSV, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset)) # 一次性加载所有验证数据
    val_images, val_bmis = next(iter(val_loader))
    val_images, val_bmis = val_images.to(device), val_bmis.to(device)
    
    predictions = model(val_images).squeeze(-1)
    
    val_loss = criterion(predictions, val_bmis).item()
    val_rmse = torch.sqrt(torch.tensor(val_loss)).item()
    val_mae = torch.abs(predictions - val_bmis).mean().item()
    
    print(f"Final Validation Stats:")
    print(f"  - Val Loss: {val_loss:.4f}")
    print(f"  - Val RMSE: {val_rmse:.4f}")
    print(f"  - Val MAE: {val_mae:.4f}")