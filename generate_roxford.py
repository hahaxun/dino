from PIL import Image
import requests
from io import BytesIO
import os
import pickle
data_images = open('data.txt', 'r')
Lines = data_images.readlines()

saved_pkl = {"gnd_fname":"hahaxun","dir_data":".","imlist":Lines[800000:1000000],"qimlist":Lines[800000:800001],}
with open('hahaxun.pickle', 'wb') as handle:
    pickle.dump(saved_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

print([[Lines[i] for i in [41,42,56711,  89892, 128781,  57874, 185897,    767,  92655, 103707,   9594]]])
'''
local_rank = (int(os.environ["LOCAL_RANK"]) + 1)

i = 0
for line in Lines:
    i = i + 1
    if i % local_rank != 0:
        continue
    try:
        url = line.split(" ")[0]
        print(url)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB').resize((800,800))
        url = url.split("/")[-1]
        img.save("Images/%s.jpeg"%url)
    except Exception as e:
        print(e)'''
