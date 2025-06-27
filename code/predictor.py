import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN

# 1. 定义和训练时完全一样的模型结构
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

class BMIPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. 初始化模型并加载权重
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # 设置为评估模式
        
        # 3. 初始化人脸检测器
        self.mtcnn = MTCNN(keep_all=False, device=self.device, min_face_size=40)
        
        # 4. 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("BMIPredictor initialized successfully!")

    def predict(self, image_stream):
        """
        接收图片的流对象，返回预测的BMI值或错误信息。
        """
        try:
            # 直接用 Image.open 打开流
            img = Image.open(image_stream).convert('RGB')
        except Exception as e:
            # 捕获 Pillow 解析错误
            return None, f"Could not open image. It might be corrupted or in an unsupported format. Details: {e}"

        # 1. 人脸检测和裁剪
        face_tensor = self.mtcnn(img)
        if face_tensor is None:
            return None, "No face detected in the image."
        
        # 将裁剪出的人脸tensor转回PIL Image，以便应用transform
        face_pil = transforms.ToPILImage()(face_tensor)

        # 2. 图像预处理
        img_tensor = self.transform(face_pil).unsqueeze(0).to(self.device) # 增加一个batch维度
        
        # 3. 模型预测
        with torch.no_grad():
            prediction = self.model(img_tensor)
            
        # 4. 返回结果
        bmi_value = prediction.item()
        return round(bmi_value, 2), None