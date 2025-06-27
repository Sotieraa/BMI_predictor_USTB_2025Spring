import pandas as pd
import os
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split # 引入划分工具

# --- 配置 ---
SOURCE_CSV = 'reddit_bmi_data_large.csv'
IMAGE_DIR = 'reddit_images_large' # 原始图片目录
PROCESSED_DIR = 'processed_faces' # 裁剪后的人脸存放目录
FINAL_CSV = 'final_dataset.csv' # 中间产物

# 最终为训练脚本准备的文件
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'

# --- 初始化 ---
os.makedirs(PROCESSED_DIR, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)

# --- 加载数据 ---
try:
    df = pd.read_csv(SOURCE_CSV)
    if '\\' in df['image_path'].iloc[0]:
        print("检测到Windows路径分隔符'\\', 正在转换为'/'...")
        df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
    
    print(f"加载了 {len(df)} 条记录，共 {df['post_id'].nunique()} 个帖子。")
except FileNotFoundError:
    print(f"错误: 原始数据CSV文件 '{SOURCE_CSV}' 未找到. 请先运行爬虫脚本。")
    exit() # 如果文件不存在，直接退出

# --- 开始处理 ---
final_data = []
processed_count = 0
failed_count = 0

# 按 post_id 分组处理
for post_id, group in tqdm(df.groupby('post_id'), desc="处理帖子"):
    image_path = group['image_path'].iloc[0]
    
    if not os.path.exists(image_path):
        failed_count += group.shape[0]
        continue

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        failed_count += group.shape[0]
        continue

    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        failed_count += group.shape[0]
        continue
    
    if len(boxes) == 2:
        boxes = sorted(boxes, key=lambda b: b[0])
        face_left_box, face_right_box = boxes[0], boxes[1]

        row_before = group[group['type'] == 'before']
        row_after = group[group['type'] == 'after']

        if not row_before.empty and not row_after.empty:
            face_left = img.crop(face_left_box)
            save_path_before = os.path.join(PROCESSED_DIR, f"{post_id}_before.jpg")
            face_left.save(save_path_before)
            final_data.append({'image': save_path_before, 'bmi': row_before['bmi'].iloc[0]})
            processed_count += 1
            
            face_right = img.crop(face_right_box)
            save_path_after = os.path.join(PROCESSED_DIR, f"{post_id}_after.jpg")
            face_right.save(save_path_after)
            final_data.append({'image': save_path_after, 'bmi': row_after['bmi'].iloc[0]})
            processed_count += 1
        else:
            failed_count += group.shape[0]

    elif len(boxes) == 1:
        row_after = group[group['type'] == 'after']
        if not row_after.empty:
            face_box = boxes[0]
            face = img.crop(face_box)
            save_path = os.path.join(PROCESSED_DIR, f"{post_id}_after.jpg")
            face.save(save_path)
            final_data.append({'image': save_path, 'bmi': row_after['bmi'].iloc[0]})
            processed_count += 1
            failed_count += 1 # 丢弃了 before 的数据
        else:
            failed_count += group.shape[0]
    
    else:
        failed_count += group.shape[0]

print(f"\n处理完成！")
print(f"成功处理并保存了 {processed_count} 张人脸。")
print(f"因各种原因，丢弃了 {failed_count} 条记录。")

# --- 保存并划分数据集 ---
if final_data:
    final_df = pd.DataFrame(final_data)
    # 重命名列以匹配训练脚本
    final_df = final_df.rename(columns={'face_path': 'image'})
    final_df.to_csv(FINAL_CSV, index=False)
    print(f"完整数据集已保存到: {FINAL_CSV} (共 {len(final_df)} 条)")

    # --- 自动划分为训练集和验证集 (80/20) ---
    print("\n正在将数据集划分为训练集和验证集...")
    train_df, val_df = train_test_split(final_df, test_size=0.2, random_state=42)
    
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    
    print("划分完成！")
    print(f"训练集 ({len(train_df)}条) 已保存到: {TRAIN_CSV}")
    print(f"验证集 ({len(val_df)}条) 已保存到: {VAL_CSV}")
else:
    print("没有生成任何有效数据，请检查原始图片和代码。")