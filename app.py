from flask import Flask, render_template, request, redirect
import cv2
import os

from utils import ImageProcessor
from ScanInvoice import scan_document
from Extract_Info import extraction, writeCSV


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/images/'
app.config['CSV_FOLDER'] = './static/csv/'
app.secret_key = 'mykeys'

def extract(image_path, csv_path):
    image = cv2.imread(image_path)
    imgScan, device = scan_document(image)
    imgScan = ImageProcessor.convert_to_binar(imgScan)
    info, total, prod = extraction(imgScan, device)
    writeCSV(csv_path, info, total, prod)
    return info, total, prod.to_dict(orient='records')

@app.route('/')
def home():
    return render_template('predict.html')
@app.route('/', methods = ['GET','POST'])
def upload():
    file = request.files['imgfile']
    if file:
        path = app.config['UPLOAD_FOLDER'] + file.filename
        file.save(path)
        csv_path = app.config['CSV_FOLDER'] + os.path.splitext(file.filename)[0] + '.csv'
        info, total, prod = extract(path, csv_path)
        information = [info, total, prod]
        return render_template('predict.html', filepath = path,
                               order_info = information, output_path = csv_path)

    return redirect(request.url)

if __name__ == "__main__":
    app.run(debug = True)