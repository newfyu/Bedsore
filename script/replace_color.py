"""将mask背景替换为0"""
from PIL import Image
import os
import concurrent.futures


def replace_color(img_name):
    img = Image.open(f'{img_dir}{img_name}')
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            r,g,b = img.getpixel((i,j))
            if r==48 and g==112 and b==32:
                r=0;g=0;b=0
                img.putpixel((i,j), (r,g,b))
    img.save(f'{img_dir}{img_name}')
    print(img_name) 

if __name__ == "__main__":
    img_dir = '../data/VOCdevkit/VOC2007/SegmentationClass/'
    img_list = [i for i in os.listdir(img_dir) if 'png' in i]
    #  for img_name in img_list:
        #  replace_color(img_name)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(replace_color, img_list)
