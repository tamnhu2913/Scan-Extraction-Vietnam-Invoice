import cv2

from utils import ImageProcessor
from ScanInvoice import scan_document
from Extract_Info import extraction, writeCSV

image_path = '.\image\image-7.jpg'
image = cv2.imread(image_path)
imgScan, device = scan_document(image)
imgScan = ImageProcessor.convert_to_binar(imgScan)
info, total, prod = extraction(imgScan, device)
writeCSV('.\image\output-image-7.csv',info, total, prod)
