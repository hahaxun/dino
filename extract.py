from PIL import Image
import requests
from io import BytesIO
import os
data_images = open('data.txt', 'r')
Lines = data_images.readlines()

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
        print(e)

