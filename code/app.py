from flask import Flask, request, render_template, jsonify
from predictor import BMIPredictor # 从刚刚创建的文件中导入类
import os

# --- 初始化 ---
app = Flask(__name__)
# 加载预测器
model_path = 'final_bmi_predictor.pth' 
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please make sure it's in the same directory as app.py.")
bmi_predictor = BMIPredictor(model_path)


# --- 路由定义 ---
@app.route('/', methods=['GET'])
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理图片上传和BMI预测"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            # 直接将文件流 file 传递给预测器
            bmi, error_msg = bmi_predictor.predict(file)
            
            if error_msg:
                return jsonify({'error': error_msg})
            else:
                return jsonify({'bmi': bmi})

        except Exception as e:
            # 捕获其他意外错误，例如 Pillow 无法解析一个非图片文件
            return jsonify({'error': f'Error processing image: {e}'}), 500

# --- 启动服务 ---
if __name__ == '__main__':
    # host='0.0.0.0' 让服务可以被局域网内的其他设备访问
    app.run(host='0.0.0.0', port=5000, debug=True)