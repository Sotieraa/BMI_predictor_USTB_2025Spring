import os

def rename_jpg_extensions(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.JPG'):
            # 构建原始文件的完整路径
            old_path = os.path.join(directory, filename)
            # 构建新文件名（替换扩展名）
            new_filename = filename[:-4] + '.jpg'
            new_path = os.path.join(directory, new_filename)
            
            # 重命名文件
            try:
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_filename}')
            except OSError as e:
                print(f'Error renaming {filename}: {e}')

# 指定要处理的目录
data_dir = './data'

# 检查目录是否存在
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    rename_jpg_extensions(data_dir)
    print("文件后缀名修改完成！")
else:
    print(f"错误：目录 {data_dir} 不存在或不是一个目录")