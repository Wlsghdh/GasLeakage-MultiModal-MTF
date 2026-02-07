import sys


import numpy as np


import pandas as pd


from PIL import Image


import os

def save_vertical_stripe_heatmap(sensor_values, save_path, filename='heatmap_stripes.png'):
    """
    sensor_values: 7개 센서값 (예: [615, 339, 396, 412, 574, 598, 312])
    각 센서값을 세로 방향으로 stripe처럼 배치하여 224x224 heatmap 생성
    """
    assert len(sensor_values) == 7, 

    width = 224
    height = 224
    stripe_width = width // 7  


    norm_vals = np.interp(sensor_values, (min(sensor_values), max(sensor_values)), (0, 255)).astype(np.uint8)


    img_array = np.zeros((height, width), dtype=np.uint8)

    for i, val in enumerate(norm_vals):
        start = i * stripe_width
        end = (i + 1) * stripe_width if i < 6 else width  
        img_array[:, start:end] = val


    img_rgb = np.stack([img_array]*3, axis=2)

    os.makedirs(save_path, exist_ok=True)
    img = Image.fromarray(img_rgb)
    img.save(os.path.join(save_path, filename))


    
sensor = [615, 339, 396, 412, 574, 598, 312]
save_vertical_stripe_heatmap(sensor, save_path='stripe_heatmaps', filename='smoke_stripe_224.png')


Nogas = sensor_data[sensor_data['Gas']=='NoGas']
for i in range(1600):
    sensor = Nogas.iloc[i,1:8]
    save_vertical_stripe_heatmap(sensor, save_path='Nogas',filename=f'{i}_NoGas.png')
    
    
Nogas = sensor_data[sensor_data['Gas']=='Mixture']
for i in range(1600):
    sensor = Nogas.iloc[i,1:8]
    save_vertical_stripe_heatmap(sensor, save_path='Mixture',filename=f'{i}_Mixture.png')
    

Nogas = sensor_data[sensor_data['Gas']=='Perfume']
for i in range(1600):
    sensor = Nogas.iloc[i,1:8]
    save_vertical_stripe_heatmap(sensor, save_path='Perfume',filename=f'{i}_Perfume.png')
    

Nogas = sensor_data[sensor_data['Gas']=='Smoke']
for i in range(1600):
    sensor = Nogas.iloc[i,1:8]
    save_vertical_stripe_heatmap(sensor, save_path='Smoke',filename=f'{i}_Smoke.png')