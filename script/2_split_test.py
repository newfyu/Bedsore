import pandas as pd

if __name__ == "__main__":
    trainval = pd.read_csv(
        'data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        names=['filename'])
    val = pd.read_csv('data/VOCdevkit/VOC2007/ImageSets/Main/val.txt',
                      names=['filename'])
    diff_df = pd.concat([trainval, val, val]).drop_duplicates(keep=False)
    diff_df.to_csv('data/VOCdevkit/VOC2007/ImageSets/Main/train.txt',
                   index=None,
                   header=None)
