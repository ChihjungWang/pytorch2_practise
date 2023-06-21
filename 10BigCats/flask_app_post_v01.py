import requests as req

# 向伺服器發送請求
def demo (url, files, data=None):
    result = req.post(url, data=data, files=files).text
    return result

if __name__ == "__main__":
    url = "http://127.0.0.1:5000/image_classification"
    # 使用二進位形式打開圖片
    files = {"image": open("kaggle/10BigCats/test/LIONS/1.jpg","rb")}
    data = {"delete_file":True}
    # 將圖片和參數一併上傳
    r = demo(url, files, data)
    print(r)