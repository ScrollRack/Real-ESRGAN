# import argparse
import cv2
import glob
import os
import requests
import time
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def uploadResults(output):
    # Upload results to S3
    pass

def callWebhook():
    # Call webhook to notify of completion
    pass

def main():
    model_name = 'RealESRGAN_x4plus'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    tile = 0
    tile_pad = 10
    pre_pad = 0
    fp32 = True
    gpu_id = 1
    outscale = 4

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    output_path = 'outputs'
    os.makedirs(output_path, exist_ok=True)

    input = 'https://scrollrack-image-generator.s3.us-east-2.amazonaws.com/1002_822612_f21a50702d5c4a9cbc1a931350053da1.png'
    data = requests.get(input)
    img = cv2.imdecode(np.frombuffer(data.content, np.uint8), cv2.IMREAD_UNCHANGED)
    img_mode = None
    img_name = input.split('/')[-1]
    save_path = os.path.join(output_path, f'{img_name}')

    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set tile with a smaller number.')
    else:
        cv2.imwrite(save_path, output)

while True:
    print('Starting worker')

    main()

    time.sleep(0.5)
