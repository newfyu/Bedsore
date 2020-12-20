import demjson

# 项目中使用的一些零散有用函数等

#  def clear_json(file):
    #  """
    #  读取coco的json文件，删除没有标注的图片，生成新的json
    #  """
    #  json = demjson.decode_file(file)
    #  ids = []
    #  for i in val['annotations']:
        #  ids.append(i['image_id'])
    #  ids = set(ids)
    #  num_image = len(json['images'])
    #  no_annotations_id = set(list(range(1,num_image+1))) - ids
    #  for i in json['images']:
        #  id = i['id']
        #  if id in no_annotations_id:
            #  json['images'].remove(i)
    #  demjson.encode_to_file(file[:-5]+'_clearning.json', val)

def collate_fn(batch):
    #  print(batch)
    return tuple(zip(*batch))
