import sys
import argparse
import cv2

sys.path.append('../../umikry-core/')
from UmikryCore import umikry

parser = argparse.ArgumentParser(description='umikry script to pseudonymise unknown people on images')
parser.add_argument('image', metavar='IMAGE', type=str,
                    help='the input photo for the umikry transformation')
parser.add_argument('--detection', default='CAFFE',
                    help='umikry detection method (HAAR, CAFFE, CNN)')
parser.add_argument('--transformation', default='GAN',
                    help='umikry transformation method (GAN, AUTOENCODER, BLUR)')
parser.add_argument('--output', default='output.jpg',
                    help='output file path')

args = parser.parse_args()

image = cv2.imread(args.image)
image = umikry(image, detection=args.detection, transformation=args.transformation)

cv2.imwrite(args.output, image)
