import cv2
import re
import numpy as np
import torch
import torchvision
from datetime import datetime
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class ImageProcessor:
    @staticmethod
    def get_bbox(edges, image):
        bbox = np.zeros(edges.shape, dtype=np.float32)

        s = np.sum(edges, axis=1)
        bbox[0] = edges[np.argmin(s)]
        bbox[2] = edges[np.argmax(s)]

        d = np.diff(edges, axis=1)
        bbox[1] = edges[np.argmin(d)]
        bbox[3] = edges[np.argmax(d)]
        return bbox

    @staticmethod
    def define_newbbox(bbox):
        tl, tr, br, bl = bbox
        widthTop = np.sqrt(np.sum((tr - tl) ** 2))
        widthBottom = np.sqrt(np.sum((bl - br) ** 2))
        maxWidth = max(int(widthTop), int(widthBottom))

        heightLeft = np.sqrt(np.sum((tl - bl) ** 2))
        heightRight = np.sqrt(np.sum((tr - br) ** 2))
        maxHeight = max(int(heightLeft), int(heightRight))
        newPoint = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype=np.float32)
        return newPoint, maxWidth, maxHeight

    def warp_image(self, image, box):
        bbox = self.get_bbox(box, image)
        newBbox, maxWidth, maxHeight = self.define_newbbox(bbox)
        matrix = cv2.getPerspectiveTransform(bbox, newBbox)
        warpedImage = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
        return warpedImage

    @staticmethod
    def convert_to_binar(image):
        image = cv2.detailEnhance(image, sigma_s=20, sigma_r=0.15)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image

class ScanDocument:
    def __init__(self, device):
        self.device = device

    def model_deeplabv(self, cppath):
        model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=2)
        model = model.to(self.device)
        checkpoint = torch.load(cppath, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        return model.eval()

    def preprocess_image(self, image, mean=[0.4611, 0.4359, 0.3905], std=[0.2193, 0.2150, 0.2109]):
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor

    @staticmethod
    def get_contours(segments):
        canny = cv2.Canny(segments.astype(np.uint8), 225, 255)
        canny = cv2.dilate(canny, np.ones((5, 5), dtype=np.uint8), iterations=1)
        contour, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contour, key=cv2.contourArea, reverse=True)[0]
        epsilon = 0.02 * cv2.arcLength(contour, True)
        edges = cv2.approxPolyDP(contour, epsilon, True)
        return edges

    @staticmethod
    def extended_image(image, pad):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image_extended = np.zeros((image.shape[0] + pad * 2, image.shape[1] + pad * 2, image.shape[2]), dtype=np.uint8)
        image_extended[pad: pad + image.shape[0], pad: pad + image.shape[1]] = image
        return image_extended

    @staticmethod
    def remove_center(approx):
        x, y = (np.max(approx, axis=0) - np.min(approx, axis=0)) // 2
        xmin, ymin = min(approx[:, 0]), min(approx[:, 1])
        i = 0
        while len(approx) > 4:
            diff = np.abs(approx[i] - approx)
            close_indice = np.where((diff[:, 0] < x) & (diff[:, 1] < y) & (np.arange(len(approx)) != i))[0]
            if len(close_indice) > 0:
                k = close_indice[0]
                if approx[i, 0] < x + xmin and approx[i, 1] < y + ymin:  # Top-Left
                    approx[i] = np.min(np.vstack((approx[i], approx[k])), axis=0)
                elif approx[i, 0] >= x + xmin and approx[i, 1] < y + ymin:  # Top-Right
                    approx[i, 0] = max(approx[i, 0], approx[k, 0])
                    approx[i, 1] = min(approx[i, 1], approx[k, 1])
                elif approx[i, 0] < x + xmin and approx[i, 1] >= y + ymin:  # Bottom-Left
                    approx[i, 0] = min(approx[i, 0], approx[k, 0])
                    approx[i, 1] = max(approx[i, 1], approx[k, 1])
                else:  # Bottom-Right
                    approx[i] = np.max(np.vstack((approx[i], approx[k])), axis=0)
                approx = np.delete(approx, k, axis=0)
                i -= 1
            i += 1
        return approx


class ExtractorInfor:
    def __init__(self, device):
        self.device = device

    def model_vietocr(self, cppath):
        """
          device: 'cuda:0' or 'cuda:1' for gpu and 'cpu'
        """
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = cppath
        config['device'] = self.device

        return Predictor(config)

    @staticmethod
    def check(i, tam, regex):
        pattern = ' '.join(t for t in tam)
        if re.search(regex, pattern, re.IGNORECASE):
            return i
        return -1

    def sort_and_group_texts(self, boxes, texts):
        sl = r'(sl)|(số lượng)'
        tt = r't(?![aâăáàãảạấầẫẩậắằẵẳặ])\w{1}ng|cộng|th[àa]nh [tiền|toán]+'
        Y = np.min(boxes[:, :, 1], axis=1)
        diff = Y[1:] - Y[:-1]
        sort_value = {'box': [], 'text': [], 'split': []}
        flag = np.array([0, 0])
        temp = {'box': [boxes[0]], 'text': [texts[0]]}
        count = 0
        for i in range(len(diff)):
            if diff[i] < np.mean(diff):
                temp['box'].append(boxes[i + 1])
                temp['text'].append(texts[i + 1])
            else:
                n = len(sort_value['text'])
                if (c := self.check(n, temp['text'], sl)) != -1 and flag[0] == 0:
                    sort_value['split'].append(n)
                    flag[0] = 1
                elif (c := self.check(n, temp['text'], tt)) != -1 and flag[1] == 0 and flag[0] == 1:
                    sort_value['split'].append(n)
                    flag[1] = 1
                box = np.array(temp['box'])
                idx = np.argsort(np.min(box[:, :, 0], axis=1))
                if re.search(r'^\d{1}$', temp['text'][idx[0]]):
                    temp['box'] = [temp['box'][i] for i in range(len(temp['box'])) if i != idx[0]]
                    temp['text'] = [temp['text'][i] for i in range(len(temp['text'])) if i != idx[0]]
                else:
                    temp['box'] = [temp['box'][i] for i in idx]
                    temp['text'] = [temp['text'][i] for i in idx]

                sort_value['box'].append(temp['box'])
                sort_value['text'].append(temp['text'])
                temp = {'box': [boxes[i + 1]], 'text': [texts[i + 1]]}

        if flag[1] == 0:
            n = len(sort_value['text'])
            sort_value['split'].append(n)
        sort_value['box'].append(temp['box'])
        sort_value['text'].append(temp['text'])

        return sort_value

    @staticmethod
    def sort_product(prod):
        k = []
        for i in range(len(prod['box']) - 1):
            if len(prod['box'][i]) > 1 and len(prod['box'][i + 1]) == 1:
                Max = np.max(np.array(prod['box'][i]), axis=(0, 1))[1]
                box2 = np.array(prod['box'][i + 1][0])
                Min = np.min(box2[:, 1])
                if Max - Min > 2:
                    if re.search(r'[a-zA-Z]', prod['text'][i + 1][0]):
                        prod['text'][i] = prod['text'][i] + prod['text'][i + 1]
                    else:
                        prod['text'][i] = prod['text'][i + 1] + prod['text'][i]
                    k.append(i + 1)
            return [prod['text'][i] for i in range(len(prod['text'])) if i not in k]

    @staticmethod
    def extractNameDate(info):
        extrct = {'Ten': '', 'Ngay': ''}
        keywords = ["siêu thị", "cửa hàng", "mart", "food", "market", "nhà hàng", "coffee", "cafe"]
        regex1 = '|'.join(keywords)

        regex2 = r"[ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ]"
        extrct['Ten'] = next(
            (t1 for t in info for t1 in t if
             not re.search(regex2, t1, re.IGNORECASE) or re.search(regex1, t1, re.IGNORECASE)),
            info[0][0]
        )
        for t in info:
            match = re.findall(r'\d{2}[-/\.]+\d{2}[-/\.]+\d{4}', ' '.join(t))
            if match:
                extrct['Ngay'] = re.sub(r'[^\d]', '/', match[0])
                if extrct['Ngay'][0] == '4' or extrct['Ngay'][0] == '7':
                    extrct['Ngay'] = '1' + extrct['Ngay'][1:]
                break
        extrct['Ngay'] = datetime.strptime(extrct['Ngay'], '%d/%m/%Y').date()
        return extrct

    @staticmethod
    def extractTotalSale(total):
        extrct = {'Gia': 0, 'Giam gia': 0, 'Thanh tien': 0}
        regex1 = r'\d{1,3}(?:[.,\s]\d{3})+'
        regex2 = r'\d{1,3}(?:[.,\s]\d{1,3})*'
        money, km = [], '0'
        for sub in total:
            t = [re.findall(regex1, i)[0] for i in sub if re.findall(regex1, i)]
            if re.search(r'vat|thue', ' '.join(sub), re.IGNORECASE) and t:
                continue
            elif re.search(r'giam|giảm', ' '.join(sub), re.IGNORECASE):
                km = re.findall(regex2, ' '.join(sub))[0]
            elif len(money) < 2 and t:
                money.append(int(re.sub(r'[^\d]', '', t[0])))

            if money:
                extrct['Gia'], extrct['Thanh tien'] = max(money), min(money)
            if km != "0":
                extrct['Giam gia'] = extrct['Gia'] - extrct['Thanh tien']
            else:
                extrct['Giam gia'] = int(re.sub(r'[^\d]', '', km)) if km.isdigit() else 0
        return extrct

    @staticmethod
    def extractProduct(prod):
        lst = {'Ten': None, 'SL': None, 'Gia': None, 'Thanh Tien': None}
        tien, name, sl = [], [], []
        ds = []
        for p2 in prod[1:]:
            word = [p1 for p1 in p2[:] if re.findall(r'[a-zA-Z]', p1)]
            number = [p1 for p1 in p2 if p1 not in word]
            # print(word, number)
            if word:
                for i in word[:]:
                    if re.findall(r"^\d+(?:[.,\s]\d+)*(?: (b[iị]ch|gói|chai|kg|m[óớ]|bó|cuộn|cá[in]|hộp|lốc|lon))$", i,
                                  re.IGNORECASE):
                        n = re.findall(r'^[\d,.]+', i)
                        if n:
                            sl.append(float(re.sub(r'[^\d]', '.', n[0])))
                    elif not re.findall(r'\b(?:b[iị]ch|gói|chai|kg|m[óớ]|bó|cuộn|cá[in]|hộp|lốc|lon)\b', i,
                                        re.IGNORECASE):
                        name.append(re.sub('[;+-]|(^\d+)', '', i))
                lst['Ten'] = ' '.join(name)

            if number:
                for i in number[:]:
                    if re.findall(r'\d{1,3}(?:[.,\s]\d{3})+', i):
                        n = re.findall(r'\d{1,3}(?:[.,\s]\d{3})+', i)[0]
                        tien.append(int(re.sub(r'[^\d]', '', n)))
                        number.remove(i)
                    elif i:
                        sl.append(float(re.sub(r'[^\d]', '.', i)))
                        number.remove(i)
                if len(tien) > 1:
                    lst['Gia'] = tien[-2]
                    lst['Thanh Tien'] = tien[-1]
                else:
                    lst['Thanh Tien'] = lst['Gia'] = tien[-1]

                if sl:
                    sl = [i for i in sl if i != 0 and i < 1000]
                lst['SL'] = sl[-1] if sl else lst['Thanh Tien'] / lst['Gia']

            if all(lst.values()):
                ds.append(lst)
                lst = {'Ten': None, 'SL': None, 'Gia': None, 'Thanh Tien': None}
                tien, name, sl = [], [], []
        return ds