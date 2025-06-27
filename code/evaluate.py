import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

# --- 1. 定义模型结构 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(256), nn.MaxPool2d(2),
        )
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

# --- 2. 配置 ---
TEST_ANNOTATION_CSV = 'test/annotation_fixed.csv'
TEST_IMAGE_DIR = 'test/data'
MODEL_PATH = 'final_bmi_predictor.pth'

BATCH_SIZE = 64
NUM_WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 定义数据集和加载器 (简化版) ---
class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # 读取并清理 annotation_fixed.csv
        try:
            # 只读取我们需要的列，并直接命名，跳过表头
            self.df = pd.read_csv(csv_file, usecols=[0, 3], names=['image', 'bmi'], header=0)
            self.df.dropna(inplace=True)
        except FileNotFoundError:
            print(f"错误: 测试集标注文件 '{csv_file}' 未找到。")
            exit()
            
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        img_name = self.df.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"警告: 测试图片 {img_path} 未找到，将跳过。")
            return None, None # 返回None，由collate_fn处理
            
        bmi = torch.tensor(float(self.df.iloc[index, 1]), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, bmi

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

# 使用和验证集完全相同的图像变换
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("正在加载测试集...")
test_dataset = TestDataset(TEST_ANNOTATION_CSV, TEST_IMAGE_DIR, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
print(f"测试集加载完成，共 {len(test_dataset)} 个样本。")


# --- 4. 加载模型并进行评估 ---
print("\n" + "="*20 + " 开始在测试集上评估 " + "="*20)

# 初始化模型结构
model = SimpleCNN().to(device)

# 加载训练好的权重
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"成功加载模型权重从: {MODEL_PATH}")
except FileNotFoundError:
    print(f"错误: 模型文件 '{MODEL_PATH}' 未找到。")
    exit()

# 设置为评估模式
model.eval()

# 使用 SmoothL1Loss 来计算损失，与训练时保持一致
criterion = nn.SmoothL1Loss() 

all_predictions = []
all_targets = []

print("正在测试集上进行预测...")
with torch.no_grad():
    for data, targets in tqdm(test_loader):
        # 跳过空的batch (如果collate_fn返回了空tensor)
        if data.nelement() == 0: continue
            
        data, targets = data.to(device), targets.to(device)
        scores = model(data).squeeze(-1)
        all_predictions.append(scores)
        all_targets.append(targets)
# 将所有batch的预测和目标拼接起来
all_predictions = torch.cat(all_predictions)
all_targets = torch.cat(all_targets)

# --- 5. 计算并打印最终性能指标 ---
final_loss = criterion(all_predictions, all_targets).item()
final_rmse = torch.sqrt(nn.functional.mse_loss(all_predictions, all_targets)).item()
final_mae = torch.abs(all_predictions - all_targets).mean().item()

print("\n" + "="*20 + " 最终评估结果 " + "="*20)
print(f"  - 测试集损失 (SmoothL1Loss): {final_loss:.4f}")
print(f"  - 测试集均方根误差 (RMSE):    {final_rmse:.4f}")
print(f"  - 测试集平均绝对误差 (MAE):    {final_mae:.4f}")
print("="*50)