import streamlit as st
import torchvision.transforms as T
from PIL import Image
import time
import gdown
import os

from model import MyFasterRCNN
from utils import draw_bbox

SZ = 512
CKPT_PATH = 'model.ckpt'

col1, col2, col3 = st.columns([1, 10, 1])


@st.cache
def load_model():
    if not os.path.exists(CKPT_PATH):
        url = 'https://drive.google.com/uc?id=1k6wQ27b0heo3Wowv7lGJor_bkG2GCBcE'
        st.spinner('下载模型')
        gdown.download(url, CKPT_PATH, quiet=False)
    else:
        st.spinner('已经下载模型')


load_model()


with col1:
    st.write('')

with col3:
    st.write('')

with col2:
    lang = st.radio('', ('中文', 'English'))
    if lang == '中文':
        title_text = '压力性损伤自动分期系统'
        wait_text1 = '等待上传图片'
        wait_text2 = '正在分析，请稍等……'
        wait_text3 = '分析完毕'
        st.sidebar.markdown("""
# 图例
1 1期    
2 2期    
3 3期    
4 4期    
5 不可分期    
6 深部组织损伤    
:full_moon: 腐肉    
:red_circle: 肉芽    
:large_blue_circle: 坏死    
""")
    else:
        title_text = 'Automatic Staging System for Pressure Injury'
        wait_text1 = 'Waiting to upload pictures'
        wait_text2 = 'Analyzing, please wait……'
        wait_text3 = 'Analysis complete'
        st.sidebar.markdown("""
# Legend
1 Stage1    
2 Stage2    
3 Stage3    
4 Stage4    
5 Unstageable    
6 Deep Tissue Pressure Injury    
:full_moon: Slough    
:red_circle: Granulation    
:large_blue_circle: Necrotic    
""")
    st.title(title_text)


# load model
#  model = MyFasterRCNN.load_from_checkpoint('mlruns/3/8125991a77df4bf9a5499367b41e3970/checkpoints/epoch=59-step=16979.ckpt')
    model = MyFasterRCNN.load_from_checkpoint(CKPT_PATH)
    net = model.net
    net.eval()

# wait image
    uploaded_file = st.file_uploader('', type=['jpg', 'png', 'jpeg', 'bmp'])
    wait_msg = st.empty()
    if not uploaded_file:
        wait_msg.text(wait_text1)
        st.stop()

    wait_msg.text(wait_text2)

    pred_img = Image.open(uploaded_file)
    t = int(time.time())
    #  pred_img.save(f'upload/{t}.png') # save upload image

    x = T.ToTensor()(pred_img)
    out = net([x], None)[0]
    out_image = draw_bbox(x, out, th=0.5).resize((SZ, SZ))
    #  out_image.save(f'upload/{t}_out.png') # save output image

    st.image(pred_img.resize((SZ, SZ)))
    st.image(out_image)
    wait_msg.text(wait_text3)
    uploaded_file = None
