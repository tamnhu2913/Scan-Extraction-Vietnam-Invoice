from utils import *
import torch
import cv2
import matplotlib.pyplot as plt
import os

def scan_document(image, resizeShape = 383, pad = 50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_deeplabv('.\checkpoint\model_r50_iou_mix_2C020.pth', device)
    height, width, C = image.shape
    resizeImage = cv2.resize(image, (resizeShape, resizeShape))
    tensorImage = prepocess_image(resizeImage, device)
    with torch.no_grad():
        predict = model(tensorImage)['out'][0]

    predict = predict.cpu().argmax(0).numpy() * 255
    predict = cv2.resize(predict.astype(np.uint8), (width, height),interpolation=cv2.INTER_LINEAR)

    predict = extended_image(predict, pad).squeeze()
    extendedImage = extended_image(image, pad)

    bbox = np.array(get_contours(predict)).reshape((-1,2))
    draw = cv2.drawContours(extendedImage.copy(), [bbox], -1, (0,255,0), 2)
    if bbox.shape[0] > 4:
        bbox = remove_center(bbox)
    imgScan = warpImage(extendedImage, bbox)
    return imgScan

image_path = '.\image\image-9.jpg'
image = cv2.imread(image_path)
imgScan = scan_document(image)
imgScan = prepocess_ocr(imgScan)
save_path = os.path.splitext(image_path)[0] + '_scan' + os.path.splitext(image_path)[1]
cv2.imwrite(save_path, imgScan)

cv2.waitKey(0)
cv2.destroyAllWindows()