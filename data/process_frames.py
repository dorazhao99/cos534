# Adapted from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
# Aligns face images and crops into square image.

import argparse
import os
import cv2
import dlib
import imutils
import numpy as np
from tqdm import tqdm
from face_aligner import FaceAligner, get_avg_lightness

def rect_to_bb(rect):
    x, y = rect.left(), rect.top()
    w, h = rect.right() - x, rect.bottom() - y
    return (x, y, w, h)

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, default="CC_images")
parser.add_argument('--outdir', type=str, default="CC_aligned")
parser.add_argument('--threshold', type=float, default=0.0)
arg = vars(parser.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

# aligns the face
for z in tqdm(os.listdir(arg['indir'])):
    if not z.startswith('.'):
        filedir = '{}/{}'.format(arg['indir'], z)
        if not os.path.exists('{}/{}'.format(arg['outdir'], z)):
            os.makedirs('{}/{}'.format(arg['outdir'], z))
            
        for img in os.listdir(filedir):
            if len(img) < 4 or img[-4:] != '.png':
                continue
            image = cv2.imread('{}/{}'.format(filedir, img))
            image = imutils.resize(image, width=1000)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 2)
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
                faceAligned = fa.align(image, gray, rect)
                avg_l = get_avg_lightness(faceAligned)
                if avg_l > arg['threshold']:
                    # print('Saved {}'.format(img))
                    cv2.imwrite('{}/{}/{}'.format(arg['outdir'], z, img), faceAligned)
                else:
                    print('Too dark: {:.2f}'.format(avg_l))
                break