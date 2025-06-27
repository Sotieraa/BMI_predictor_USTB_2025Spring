import praw
import re
import requests
import os
import pandas as pd
from tqdm import tqdm
import time

# --- 配置 ---
CLIENT_ID = 'xf9D9oIEjse5oco8130zsg'
CLIENT_SECRET = 'QlfnQadrJkzZTh8S9DCTRUATld9vpA'
USER_AGENT = 'bmi_scraper/0.1 by u/Forsaken-War-9479'

SUBREDDIT_NAME = 'progresspics'
POST_LIMIT_PER_CATEGORY = 2000
TARGET_DATA_POINTS = 1000
IMAGE_DIR = 'reddit_images'
CSV_PATH = 'reddit_bmi_data.csv'

# --- 创建目录 ---
os.makedirs(IMAGE_DIR, exist_ok=True)

def parse_title_robust(title):
    """
    标题解析器
    尝试匹配多种格式的身高和体重
    """
    title = title.lower() # 转换为小写以便匹配

    # --- 解析身高 ---
    height_str, height_unit = None, None
    # 匹配英制: 5'11" or 5' 11"
    height_match_ft = re.search(r'(\d)\'\s*(\d{1,2})\"?', title)
    if height_match_ft:
        feet = height_match_ft.group(1)
        inches = height_match_ft.group(2)
        height_str = f"{feet}'{inches}\""
        height_unit = 'ft'
    else:
        # 匹配公制: 180cm or 180 cm
        height_match_cm = re.search(r'(\d{3})\s*cm', title)
        if height_match_cm:
            height_str = height_match_cm.group(1)
            height_unit = 'cm'
    
    if not height_str:
        return None

    # --- 解析体重 ---
    # 核心格式: [150lbs > 130lbs] or [150 > 130] or [80kg > 70kg]
    # 更灵活, 允许缺少单位, 允许浮点数
    weight_match = re.search(
        r'\[\s*(\d+\.?\d*)\s*(?:lbs|kg)?\s*>\s*(\d+\.?\d*)\s*(?:lbs|kg)?', 
        title
    )
    if not weight_match:
        # 尝试另一种常见格式: 200lbs -> 150lbs
        weight_match = re.search(
            r'(\d+\.?\d*)\s*(?:lbs|kg)\s*->\s*(\d+\.?\d*)\s*(?:lbs|kg)?',
            title
        )

    if not weight_match:
        return None
        
    before_weight_str = weight_match.group(1)
    after_weight_str = weight_match.group(2)

    # 判断体重单位
    # 如果标题中明确出现 'kg', 则认为是kg，否则默认为lbs
    weight_unit = 'kg' if 'kg' in title else 'lbs'

    return (height_str, before_weight_str, after_weight_str, height_unit, weight_unit)

# calculate_bmi 函数
def calculate_bmi(height_str, weight_str, height_unit, weight_unit):
    try:
        height_m = 0
        if height_unit == 'ft':
            parts = re.findall(r'\d+', height_str)
            feet = float(parts[0]) if len(parts) > 0 else 0
            inches = float(parts[1]) if len(parts) > 1 else 0
            height_m = (feet * 12 + inches) * 0.0254
        elif height_unit == 'cm':
            height_m = float(height_str) / 100

        if height_m < 1.0 or height_m > 2.5: # 过滤不合理的身高
            return None

        weight_kg = float(weight_str)
        if weight_unit == 'lbs':
            weight_kg = weight_kg * 0.453592

        if weight_kg < 30 or weight_kg > 300: # 过滤不合理的体重
            return None
            
        bmi = weight_kg / (height_m ** 2)
        if 10 < bmi < 60: # 稍微放宽BMI范围
            return round(bmi, 2)
        else:
            return None
    except (ValueError, IndexError, TypeError):
        return None

def scrape_progress_pics():
    print("正在连接到 Reddit...")
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    print(f"成功连接到 r/{SUBREDDIT_NAME}。")

    data_for_csv = []
    processed_posts = set() # 用来存储处理过的帖子ID，防止重复

    # 定义要爬取的帖子类别
    categories = {
        'Top (All Time)': subreddit.top(time_filter='all', limit=POST_LIMIT_PER_CATEGORY),
        'Hot': subreddit.hot(limit=POST_LIMIT_PER_CATEGORY),
        'New': subreddit.new(limit=POST_LIMIT_PER_CATEGORY),
    }

    with tqdm(total=TARGET_DATA_POINTS, desc="收集数据点") as pbar:
        for category_name, posts in categories.items():
            if len(data_for_csv) >= TARGET_DATA_POINTS:
                break
            
            print(f"\n开始爬取 '{category_name}' 类别...")
            for post in posts:
                # 检查是否已达到目标
                if len(data_for_csv) >= TARGET_DATA_POINTS:
                    break

                # 跳过已处理的帖子和非图片帖
                if post.id in processed_posts or post.is_self or not post.url.lower().endswith(('jpg', 'jpeg', 'png')):
                    continue
                
                parsed_data = parse_title_robust(post.title)
                
                if parsed_data:
                    height_str, before_weight, after_weight, height_unit, weight_unit = parsed_data
                    
                    bmi_before = calculate_bmi(height_str, before_weight, height_unit, weight_unit)
                    bmi_after = calculate_bmi(height_str, after_weight, height_unit, weight_unit)

                    if bmi_before and bmi_after:
                        try:
                            # 下载图片 (设置超时)
                            response = requests.get(post.url, timeout=10)
                            response.raise_for_status()

                            file_extension = post.url.split('.')[-1].split('?')[0] # 处理URL参数
                            image_filename = f"{post.id}.{file_extension}"
                            image_path = os.path.join(IMAGE_DIR, image_filename)
                            
                            with open(image_path, 'wb') as f:
                                f.write(response.content)

                            # 记录数据和处理过的ID
                            data_for_csv.append({'image_path': image_path, 'bmi_before': bmi_before, 'bmi_after': bmi_after, 'post_id': post.id})
                            processed_posts.add(post.id)
                            
                            pbar.update(2) # 一张图提供2个数据点

                        except requests.exceptions.RequestException:
                            continue # 下载失败则跳过
                        except Exception as e:
                            print(f"发生未知错误: {e}")
                            continue

    print("\n爬取循环结束。")
    
    # 将收集到的数据转换为更适合处理的格式
    final_data = []
    for item in data_for_csv:
        # 为 before 和 after 创建独立的行
        final_data.append({'image_path': item['image_path'], 'bmi': item['bmi_before'], 'type': 'before', 'post_id': item['post_id']})
        final_data.append({'image_path': item['image_path'], 'bmi': item['bmi_after'], 'type': 'after', 'post_id': item['post_id']})

    if not final_data:
        print("未能收集到任何有效数据。请检查 API 凭证和网络。")
        return

    df = pd.DataFrame(final_data)
    df.to_csv(CSV_PATH, index=False)

    print(f"\n爬取完成！")
    print(f"成功收集到 {len(df)} 个数据点。")
    print(f"图片保存在 '{IMAGE_DIR}' 文件夹中。")
    print(f"数据索引保存在 '{CSV_PATH}' 文件中。")

if __name__ == '__main__':
    scrape_progress_pics()