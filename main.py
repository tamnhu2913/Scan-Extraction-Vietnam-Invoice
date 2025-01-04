import cv2

from utils import ImageProcessor
from ScanInvoice import scan_document
from Extract_Info import extractorInfor, writeCSV

image_path = '.\image\image-9.jpg'
image = cv2.imread(image_path)
imgScan, device = scan_document(image)
imgScan = ImageProcessor.convert_to_binar(imgScan)
info, total, prod = extractorInfor(imgScan, device)
writeCSV('.\image-9-output1.csv',info, total, prod)

