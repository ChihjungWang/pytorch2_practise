import requests
 
url = 'http://127.0.0.1:5000/predict'
 
# Set image file path
image_path = 'kaggle/10BigCats/test/LIONS/1.jpg'
 
# Read image file and set as payload
image = open(image_path, 'rb')
payload = {'image': image}
 
# Send POST request with image and get response
response = requests.post(url, data=payload)
 
print(response.text)