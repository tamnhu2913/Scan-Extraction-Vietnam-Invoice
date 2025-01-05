from torch.xpu import device

from utils import ScanDocument, ImageProcessor
import numpy as np
import torch
import cv2

def scan_document(image, resizeShape = 383, pad = 50):
    ## Crop invoice in the image using pre-training Deeplabv3 model
    ## The pre-train model is download from .....
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scd = ScanDocument(device)
    model = scd.model_deeplabv('.\checkpoint\model_r50_iou_mix_2C020.pth')
    height, width, C = image.shape
    resizeImage = cv2.resize(image, (resizeShape, resizeShape))
    tensorImage = scd.preprocess_image(resizeImage)
    with torch.no_grad():
        predict = model(tensorImage)['out'][0]
    predict = predict.cpu().argmax(0).numpy() * 255
    predict = cv2.resize(predict.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

    predict = scd.extended_image(predict, pad).squeeze()
    extendedImage = scd.extended_image(image, pad)
    bbox = np.array(scd.get_contours(predict)).reshape((-1, 2))
    if bbox.shape[0] > 4:
        bbox = scd.remove_center(bbox)
    imgScan = ImageProcessor().warp_image(extendedImage, bbox)
    return imgScan, device
