import numpy as np
import torch
import torchvision
import cv2

def model_deeplabv(CPpath, device):
    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=2)
    model = model.to(device)
    checkpoint = torch.load(CPpath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    return model.eval()

def prepocess_image(image, device, mean=[0.4611, 0.4359, 0.3905], std=[0.2193, 0.2150, 0.2109]):
  preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
  image_tensor = preprocess(image)
  image_tensor = image_tensor.unsqueeze(0).to(device)
  return image_tensor

def get_contours(segments):
    canny = cv2.Canny(segments.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, np.ones((5,5), dtype = np.uint8), iterations=1)
    contour,_= cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contour, key = cv2.contourArea, reverse = True)[0]
    epsilon = 0.02 * cv2.arcLength(contour, True)
    edges = cv2.approxPolyDP(contour, epsilon, True)
    return edges

def get_bbox(edges, image):
    bbox = np.zeros(edges.shape, dtype = np.float32)

    s = np.sum(edges, axis = 1)
    bbox[0] = edges[np.argmin(s)]
    bbox[2] = edges[np.argmax(s)]

    d = np.diff(edges, axis = 1)
    bbox[1] = edges[np.argmin(d)]
    bbox[3] = edges[np.argmax(d)]
    return bbox

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

def warpImage(image, box):
    bbox = get_bbox(box, image)
    newBbox, maxWidth, maxHeight = define_newbbox(bbox)
    matrix = cv2.getPerspectiveTransform(bbox, newBbox)
    warpdedImage = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    return warpdedImage

def extended_image(image, pad):
  if len(image.shape) == 2:
    image = np.expand_dims(image, axis = -1)
  image_extended = np.zeros((image.shape[0] + pad * 2, image.shape[1] + pad * 2, image.shape[2]), dtype = np.uint8)
  image_extended[pad : pad + image.shape[0], pad : pad + image.shape[1]] = image
  return image_extended

def remove_center(approx):
  x,y = (np.max(approx, axis = 0)-np.min(approx, axis = 0)) // 2
  xmin, ymin = min(approx[:,0]), min(approx[:,1])
  i = 0
  while len(approx) > 4:
    diff = np.abs(approx[i] - approx)
    close_indice = np.where((diff[:, 0] < x) & (diff[:, 1] < y) & (np.arange(len(approx)) != i))[0]
    if len(close_indice) > 0:
      k = close_indice[0]
      if approx[i,0] < x + xmin and approx[i,1] < y + ymin: #Top-Left
        approx[i] = np.min(np.vstack((approx[i],approx[k])),axis=0)
      elif approx[i,0] >= x + xmin and approx[i,1] < y + ymin: #Top-Right
        approx[i,0] = max(approx[i,0],approx[k,0])
        approx[i,1] = min(approx[i,1],approx[k,1])
      elif approx[i,0] < x + xmin and approx[i,1] >= y + ymin: #Bottom-Left
        approx[i,0] = min(approx[i,0],approx[k,0])
        approx[i,1] = max(approx[i,1],approx[k,1])
      else:                                                    #Bottom-Right
        approx[i] = np.max(np.vstack((approx[i],approx[k])),axis=0)
      approx = np.delete(approx, k, axis = 0)
      i -= 1
    i += 1
  return approx

def prepocess_ocr(image):
    image = cv2.detailEnhance(image, sigma_s=20, sigma_r=0.15)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image