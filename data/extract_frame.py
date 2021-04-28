import cv2
import os
import numpy as np
import argparse
from face_aligner import get_avg_lightness

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, required=True, 
                    help="The path of the .txt file containing all the files in the .zip")
parser.add_argument('--threshold', type=float, default=0.0, 
                    help="Lightness threshold to automatically filter out dark images; " +
                         "keep at 0 to avoid filtering out darker skinned facs.")
parser.add_argument('--outdir', type=str, default="CC_images", 
                    help="The directory to save extracted frames.")
arg = vars(parser.parse_args())

if not os.path.exists(arg['outdir']):
    os.makedirs(arg['outdir'])

if arg['filepath'][-1] == '/':
    arg['filepath'] = arg['filepath'][:-1]
    
file_depth = len(arg['filepath'].split('/'))
if file_depth > 2: # Ignore folders which are empty (do not contain .mp4 files)
    file = arg['filepath'].split('/')[-1].split('.')[0]
    print('Processing file:', file)

    vidcap = cv2.VideoCapture(arg['filepath'])
    success, image = vidcap.read()
    print('Read a new frame:', success)

    count = 0
    if not os.path.exists("{}/{}".format(arg['outdir'], file.split('_')[0])):
        os.makedirs("{}/{}".format(arg['outdir'], file.split('_')[0]))
    while success:
        avg_l = get_avg_lightness(image)
        # save frame as PNG file if not dark
        if avg_l > arg['threshold']:
            cv2.imwrite("{}/{}/{}.png".format(arg['outdir'], file.split('_')[0], file), image)
        else:
            print('Too dark: {:.2f}'.format(avg_l))
        count += 1
        if count >= 1:
            break
        success, image = vidcap.read()
        print('Read a new frame:', success)
else:
    print('No .MP4 file, skipping folder', flush=True)

"""
for x in os.listdir('../../CasualConversationsR/'):
    filedir = '../../CasualConversationsA/{0}'.format(x)
    for i in os.listdir(filedir):
        file = i.split('.')[0]
        vidcap = cv2.VideoCapture(filedir + '/' + i)
        success,image = vidcap.read()
        count = 0
        if not os.path.exists("../../CC_images/{0}".format(file.split('_')[0])):
            os.makedirs("../../CC_images/{0}".format(file.split('_')[0]))
        while success:
            cv2.imwrite("../../CC_images/{0}/{1}.png".format(file.split('_')[0], file), image)     # save frame as JPEG file      
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
            if count > 1:
                break
"""