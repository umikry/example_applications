'''
Copyright (c) 2018, umikry.com
License AGPL-3.0

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License, version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License, version 3,
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

import sys
import argparse
import json
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
parser.add_argument('--face_store', help='output file path')
parser.add_argument('--output', default='output.jpg',
                    help='output file path')

args = parser.parse_args()
image = cv2.imread(args.image)

if args.face_store:
    with open(args.face_store, 'r') as json_file:
        familar_faces = json.load(json_file)
else:
    familar_faces = None

image = umikry(image, detection=args.detection, transformation=args.transformation,
               familar_faces=familar_faces)

cv2.imwrite(args.output, image)
