import csv

import cv2
import numpy as np
import pandas as pd
from utils import ImageProcessor, ExtractorInfor
from paddleocr import PaddleOCR
from PIL import Image

def extraction(imgScan, device):
    ## Using Paddle OCR model for bounding box detection of texts
    detectBox = PaddleOCR(use_angle_cls=True, lang='vi')
    boxes = detectBox.ocr(imgScan, rec=False)[0][::-1]
    ## Read vietnam words by pre-train Viet OCR model
    extractor = ExtractorInfor(device)
    ocr = extractor.model_vietocr('.\checkpoint/vgg_seq2seq.pth')
    boxes, texts = np.array(boxes), []
    for box in boxes:
        crop = ImageProcessor().warp_image(imgScan, box)
        crop = Image.fromarray(crop)
        text = ocr.predict(crop, return_prob=False)
        if len(text) == 1 and text in ['O', 'o', 'O%', 'o%', 'D', 'D%']:
            text = '0'
        elif text[-3:] in ['00O', '00o', '00D', '00d']:
            text = text[:-3] + '000'
        texts.append(text)
    values = extractor.sort_and_group_texts(boxes, texts)
    ## Extraction feature as name, date, number of product, ...
    info = values['text'][:values['split'][0]]
    total = values['text'][values['split'][1]:]
    prod = {'box': values['box'][values['split'][0]:values['split'][1]],
            'text': values['text'][values['split'][0]:values['split'][1]]}
    prod = extractor.sort_product(prod)
    del values

    extr_info = ExtractorInfor.extractNameDate(info)
    extr_total = ExtractorInfor.extractTotalSale(total)
    extr_prod = ExtractorInfor.extractProduct(prod)
    extr_prod = pd.DataFrame(extr_prod)

    extr_prod['Check'] = extr_prod['SL'] * extr_prod['Gia'] != extr_prod['Thanh Tien']
    if extr_total['Gia'] == extr_prod['Thanh Tien'].sum():
        extr_prod.loc[extr_prod['Check'], 'Gia'] = (extr_prod.loc[extr_prod['Check'], 'Thanh Tien'] / extr_prod.loc[extr_prod['Check'], 'SL']).astype(int)
    else:
        extr_prod.loc[extr_prod['Check'], 'Thanh Tien'] = (extr_prod.loc[extr_prod['Check'], 'Gia'] * extr_prod.loc[extr_prod['Check'], 'SL']).astype(int)

    return extr_info, extr_total, extr_prod.drop(columns=['Check'])

def writeCSV(output_path, info, total, prod):
    with open(output_path, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["Tên", info['Ten'], "Ngày", info['Ngay'].strftime('%d/%m/%Y')])

        writer.writerow(['Giá', total['Gia']])
        writer.writerow(['Giảm giá', total['Giam gia']])
        writer.writerow(['Thành tiền', total['Thanh tien']])

        writer.writerow(["Tên", "SL", "Giá", "Thành Tiền"])
        for p in prod.values:
            writer.writerow([p[0], p[1], p[2], p[3]])

    return 0

