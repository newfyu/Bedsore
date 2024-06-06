from model import MyFasterRCNN
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify
import torch
import io
import time

app = Flask(__name__)
SZ = 512
CKPT_PATH = 'model.ckpt'

model = MyFasterRCNN.load_from_checkpoint(CKPT_PATH)
model = model.net
model = model.to('cpu')
model.eval()
label_dict = {1: '1期压疮', 2: '2期压疮', 3: '3期压疮', 4: '4期压疮', 5: '不可分期压疮', 6: '深部组织损伤',7:'坏死组织',8:'腐肉',9:'肉芽组织'}

@app.route('/inference', methods=['POST'])
def inference():
    start = time.time()
    image_data = request.files['image'].read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    x = image.resize((SZ, SZ))
    x = T.ToTensor()(image)

    with torch.no_grad():
        out = model([x], None)[0]
    scores = out['scores']>0.5
    labels = out['labels']
    result = scores * labels
    # 移除result中为0的值
    result = result[result != 0].tolist()
    if len(result) == 0:
        result = "未检测到压力性损伤"
    else:
        result = [label_dict[i] for i in result]
        result = "、".join(result)
        result = f"压力性损伤自动检测系统从图片中检测到了：{result}。"
    print(f"耗时: {time.time() - start:.2f}秒")
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=5001)
