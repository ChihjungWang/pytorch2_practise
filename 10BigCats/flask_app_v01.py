from flask import Flask, jsonify, request
import logging
from werkzeug.utils import secure_filename
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch
import os
from time import ctime

app = Flask(__name__)

# 上傳的圖片保存位置
app.config['UPLOAD_FOLDER'] = "tmp/img"

# 可接受的副檔名
app.config['ALLOWED_EXTENSIONS'] = set(["png", "jpg", "jpeg"])

# 增加必要的圖片轉 Tensor 的方法
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

# 定義模型
net = resnet18()

def recognition(img_path):
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)
    result = net(img_tensor)
    label = torch.argmax(result, dim=1)
    return label

def allowed_file(filename):
    # 判斷圖片名稱是否符合需求
    return (
        "." in filename and
        filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )

# 介面的主函數
@app.route("/image_classification", methods = ["POST"])
def run(delete_file=True):
    img = request.files['image']
    if img and allowed_file(img.filename):
        # 保存上傳的圖片
        filename = secure_filename(img.filename)
        folder = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"])
        img_path = os.path.join(folder, filename)
        # 如果路徑不存在，先建立
        if not os.path.exists(folder):
            os.makedirs(folder)
        img.save(img_path)
    else:
        # 如果圖片有問題，在日誌裡記錄錯誤
        app.logger.error("Image not available")
    label = recognition(img_path)
    #  在日誌中記錄辨識結果
    app.logger.info("Result : {}".format(str(label)))
    # 辨識結束之後，可以選擇刪除臨時暫時的圖片
    if delete_file:
        os.remove(img_path)
    return str(label)


# 這個函數在每次接收到請求之前呼叫，會在記錄檔中記錄使用者IP與時間
@app.before_request
def defore_request():
    ip = request.remote_addr
    app.logger.info("Time : {} Remote ip:{}".format(ctime(), ip))

if __name__ == "__main__":
    # debug 模式下修改程式，服務會自動重新啟動
    # app.debug = True
    # 建立日誌
    handler = logging.FileHandler("flask.log")
    app.logger.addHandler(handler)
    # logger 預設只在debud模式下紀錄，但是部屬不可能用debug模式
    # 所以要記錄日誌的話，要先把日誌等級設定成 debug 等級
    app.logger.setLevel(logging.DEBUG)
    # 執行服務
    app.run(host="127.0.0.1", port=5000)




