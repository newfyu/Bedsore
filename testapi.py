import requests

# 图像文件路径
image_file_path = 'upload/1711587795.png'

# Flask应用的URL
url = 'http://localhost:5001/inference'

# 打开图像文件
image_file = open(image_file_path, 'rb')

# 构造请求
response = requests.post(url, files={'image': image_file})

# 打印响应
print(response.json())
