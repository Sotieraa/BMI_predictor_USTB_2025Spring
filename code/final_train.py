import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

# --- 1. 配置 ---
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
MODEL_SAVE_PATH = 'final_bmi_predictor.pth'

# --- 超参数 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 150
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8

# --- 2. 数据集 ---
class BMIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 加载图片失败 {img_path}, Error: {e}")
            return None, None
        bmi = torch.tensor(float(self.annotations.iloc[index, 1]), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, bmi

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

# --- 3. 数据变换与增强 ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = BMIDataset(TRAIN_CSV, transform=data_transforms['train'])
val_dataset = BMIDataset(VAL_CSV, transform=data_transforms['val'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

# --- 4. 模型定义 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(256), nn.MaxPool2d(2),
        )
        # Input 224 -> 112 -> 56 -> 28 -> 14. Output feature map size: 256 x 14 x 14
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# --- 5. 权重初始化 ---
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
model.apply(weights_init)

# --- 6. 训练组件 ---
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# --- 7. 训练与评估循环 ---
best_val_mae = float('inf')

print("="*20 + " 开始最终训练 " + "="*20)
for epoch in range(NUM_EPOCHS):
    model.train()
    running_train_loss = 0.0
    for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        data, targets = data.to(device), targets.to(device)
        scores = model(data).squeeze(-1)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * data.size(0)

    model.eval()
    running_val_loss = 0.0
    total_abs_error = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            scores = model(data).squeeze(-1)
            loss = criterion(scores, targets)
            running_val_loss += loss.item() * data.size(0)
            total_abs_error += torch.abs(scores - targets).sum().item()

    # --- 计算并打印指标 ---
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_mae = total_abs_error / len(val_loader.dataset)
    epoch_val_rmse = torch.sqrt(torch.tensor(epoch_val_loss))

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Val RMSE: {epoch_val_rmse:.4f} | "
        f"Val MAE: {epoch_val_mae:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    scheduler.step()

    # 以 Val MAE 作为保存最佳模型的标准
    if epoch_val_mae < best_val_mae:
        best_val_mae = epoch_val_mae
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"** Val MAE decreased to {best_val_mae:.4f}. Saving model to {MODEL_SAVE_PATH} **")

print("\n" + "="*20 + " 训练完成 " + "="*20)
print(f"最佳模型已保存在: {MODEL_SAVE_PATH}")
print(f"最佳验证集 MAE (Mean Absolute Error): {best_val_mae:.4f}")