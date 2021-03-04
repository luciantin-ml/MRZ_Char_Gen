import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

TEST = False


# dupla konvolucija za svaki korak u UNET-u
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # 1. konv
            nn.BatchNorm2d(out_channels),  # normalizacija
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# features je borj kanala (featura) u UNET-u za jednu stranu, druga strana ide u suprotnom smjeru po polju
# npr. za torch.Size([3, 256, 40, 40])
# 2d polje od 40x40 gdje svaki element ima polje od 256 elemenata, to su ti featuri? , svaki od tih elementata ima 3 elem. za RGB
# ulaznih kanala imamo 3 jer je rgb a izlaznih 1 jer imamo samo jednu kategoriju (2, ili je ili nije)
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if TEST == True:
                print(x[0][0].detach().numpy().shape)
                plt.imshow(x[0][1].detach().numpy(), cmap='gray')
                plt.show()

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        # for idx in range(0, len(self.ups)):
        #     x = self.ups[idx](x)
        #     skip_connection = skip_connections[idx]

        #     if TEST == True:
        #       print(x[0][0].detach().numpy().shape)
        #       plt.imshow(x[0][1].detach().numpy(), cmap='gray')
        #       plt.show()

        #     if x.shape != skip_connection.shape:
        #         x = TF.resize(x, size=skip_connection.shape[2:])

        #     concat_skip = torch.cat((skip_connection, x), dim=1)
        #     x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert x.shape == preds.shape


# TEST = True
# test()


# %%

IMAGE_HEIGHT = 160 * 3
IMAGE_WIDTH = 240 * 3

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

# %%

device = "cpu"

model = UNET(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("checkpoint_98.pth.tar", map_location=torch.device('cpu'))["state_dict"])


def predict_single(image):
    augmentations = val_transforms(image=image)
    image = augmentations["image"]
    image = torch.tensor(image, requires_grad=True).to(device)
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()
    #   torchvision.utils.save_image(preds, "./pred_100.png")
    model.train()
    return preds


# %% md

# Load model

# %%

# from google.colab import drive
#
# drive.mount('/content/drive', force_remount=True)
# # !cp drive/MyDrive/checkpoint_98.pth.tar .

# %% md

# Functions

# %%

import cv2
import imutils
from imutils import contours
import json

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

method = methods[3]


def get_text_pred(image):
    contours = get_contours(image)

    mrz_img = get_mrz_image(image, contour=contours[0])

    # plt.imshow(mrz_img)
    # plt.show()

    mrz_color = mrz_img.copy()

    mrz_img = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
    mrz_base_img = mrz_img.copy
    mrz_img = cv2.bitwise_not(mrz_img)

    # plt.imshow(mrz_img)
    # plt.show()

    mrz_img[mrz_img > 200] = 255
    mrz_img[mrz_img <= 200] = 0

    # plt.imshow(mrz_img)
    # plt.show()

    (BBs_up, BBs_down) = find_char_BB(mrz_img)

    BBs_up.sort(key=lambda x: x[0])
    BBs_down.sort(key=lambda x: x[0])

    # print(x, mrz_img.shape, len(BBs_up), len(BBs_down))

    pred_up, rois_up = pred_chars(mrz_img, BBs_up)
    pred_down, rois_down = pred_chars(mrz_img, BBs_down)

    return pred_up, pred_down, mrz_img, rois_up, rois_down


# %%

def get_mrz_image(image, contour):
    base_img = image.copy()
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # plt.imshow(base_img)
    # plt.show()

    coef_y = base_img.shape[0] / image.shape[0]
    coef_x = base_img.shape[1] / image.shape[1]

    contour[:, :, 0] = contour[:, :, 0] * coef_x
    contour[:, :, 1] = contour[:, :, 1] * coef_y

    contours_poly = cv2.approxPolyDP(contour, 3, True)
    boundRect = cv2.boundingRect(contours_poly)

    x, y, w, h = boundRect
    roi = base_img[y:y + h, x:x + w]
    box = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))

    mrz_img = un_warp(base_img, contour)

    return mrz_img


# %%

def get_roi(image, bb):
    (x, y, w, h) = bb
    roi = image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88), interpolation=cv2.INTER_LANCZOS4)
    return roi


# %%

def get_contours(image):
    pred = predict_single(image)
    pred.squeeze(0).permute(2, 1, 0).shape
    pred_img = pred.squeeze().cpu().numpy()

    pred_img[pred_img >= 0.5] = 255
    pred_img[pred_img < 0.5] = 0
    pred_img = pred_img.astype(np.uint8)

    canny_output = cv2.Canny(pred_img, 1, 255)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # plt.imshow(image)
    # plt.show()
    return contours


# %%

def un_warp(image, contour):
    img = image
    cnt = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
    rect = cv2.minAreaRect(cnt)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")

    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(img, M, (width, height))

    if warped.shape[0] > warped.shape[1]:
        rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated = warped
    return rotated


# %%

def find_char_BB(image):
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    BBs_up = []
    BBs_down = []

    for (i, c) in enumerate(cnts):
        bb = cv2.boundingRect(c)
        (x, y, w, h) = bb
        if y < image.shape[0] // 2:
            BBs_up.append(bb)
        else:
            BBs_down.append(bb)
    return (BBs_up, BBs_down)


# %%

def pred_chars(image, BB):
    chars_top = ""
    char_pred = []
    cnt = 0
    # h_img = digits[0]
    # h_img_ = digits[0]
    rois = []
    for bb in BB:
        roi = get_roi(image, bb)
        rois.append(roi)
        if np.average(roi) > 240:
            continue
        scores = []
        for idy in digits:
            res = cv2.matchTemplate(roi, digits[idy], method)
            (_, score, _, _) = cv2.minMaxLoc(res)
            scores.append(score)

        sorted = scores.copy()
        sorted.sort()
        largest = sorted[-1]
        indx = scores.index(largest)
        char_pred.append(digits_index_to_char[indx])
        # h_img = np.hstack((h_img,digits[indx]))
        # h_img_ = np.hstack((h_img_,roi))
        # cnt += 1

    return char_pred, rois


# %%

ref = cv2.imread('./OCR-B_ISO1073-2.PNG')

ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 200, 60, cv2.THRESH_BINARY_INV)[1]

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="top-to-bottom")[0]
digits = {}

for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

# %%

# for digit in digits:
#     plt.imshow(digits[digit])
#     plt.show()
digits_index_to_char = "9820765431GCIHFEDBAQORPNMLKJ<SZYXWVUT"

# %% md

# Pred Text

# %%

import cv2
import imutils
from imutils import contours
import json

# image = np.array(Image.open('./data/val_images/4.png').convert("RGB"))
#
# pred_up, pred_down, mrz_img, rois_up, rois_down = get_text_pred(image)
#
# print(pred_up)
# print(pred_down)
# plt.imshow(mrz_img)
# plt.show()

# %%

correct_img = dict()

for idx in digits_index_to_char:
    correct_img[idx] = []

# %%

from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)

from string import Template

f = open('MRZ_values.json', )
data = json.load(f)

total_c = 0
correct_c = 0

print(len(data))

for idx in tqdm(range(1000)):
    path = Template('./data/val_images/${idx}.png')
    path = path.substitute(idx=idx)
    image = np.array(Image.open(path).convert("RGB"))
    pred_up, pred_down, mrz_img, rois_up, rois_down = get_text_pred(image)

    mrz_t = data[str(idx)]['code'].split('\n')

    val_up = mrz_t[0]
    val_down = mrz_t[1]

    # print()
    # print(val_up)
    # print(pred_up)
    # print(val_down)
    # print(pred_down)
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(mrz_img)
    # plt.show()
    try:
        for idy in range(len(pred_up)):
            total_c += 1
            if pred_up[idy] == val_up[idy]:
                correct_c += 1
                correct_img[val_up[idy]].append(rois_up[idy])

        for idy in range(len(pred_down)):
            total_c += 1
            if pred_down[idy] == val_down[idy]:
                correct_c += 1
                correct_img[val_down[idy]].append(rois_down[idy])
    except:
        print('error')
        print(pred_up, pred_down, rois_up, rois_down, idx)

print(" % :")
print(correct_c / total_c)
print("Correct : ", correct_c)
print("Total : ", total_c)

# # %% md
#
# id
# for errors:
#     491, 1636, 1988


# %%
#
# !cd
# data / val_images / & & ls
#
# # %% md
#
#
# # %%
#
# !mkdir
# chars
#
# # %%
#
# !rm - rf
# chars

# %%

# plt.imshow(correct_img['9'][6])
# print(correct_img['9'][6].shape)
from string import Template

counter = 0

correct_img_dict = dict()

for idx in digits_index_to_char:
    correct_img_dict[idx] = []

path = './chars/'

template = Template('./chars/${name}.bmp')

for idx in correct_img:
    for idy in correct_img[idx]:
        counter += 1;
        correct_img_dict[idx].append(counter)
        cv2.imwrite(template.substitute(name=counter), idy)
print(counter)
with open("chars_dict.json", "w") as outfile:
    json.dump(correct_img_dict, outfile)
