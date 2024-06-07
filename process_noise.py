import os
from PIL import Image

data_dirs = ["_NumberDataSet", "_CharacterDataSet", "_SymbolDataSet"]

for data_dir in data_dirs:
    all_map = {}
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        for path in os.listdir(class_dir):
            img_path = os.path.join(class_dir, path)
            img = Image.open(img_path)

            if img.size in all_map.keys():
                all_map[img.size] += 1
            else:
                all_map[img.size] = 1
    print(all_map)

'''
{(64, 64): 46102, (28, 28): 6000}
{(64, 64): 5975, (45, 45): 63946, (28, 28): 6000}
'''



# for data_dir in data_dirs:
#     for class_name in os.listdir(data_dir):
#         class_dir = os.path.join(data_dir, class_name)
#         for path in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, path)
#             img = Image.open(img_path)
#             img_size = img.size
#             img.close()
#             if data_dir == "_CharacterDataSet" and img_size != (64, 64):
#                 os.remove(img_path)
#             elif data_dir == "_SymbolDataSet" and img_size != (45, 45):
#                 os.remove(img_path)




