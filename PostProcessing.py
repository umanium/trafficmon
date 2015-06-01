import cv2
import numpy as np

def morphological(im, operator = min, nx = 5, ny = 5):
    height, width = im.shape
    out_im = np.ones_like(im, 'uint8')
    in_pix = im
    out_pix = out_im
    for x in range(height):
        for y in range(width):
            nlst = neighbours(in_pix, width, height, x, y, nx, ny)
            if nlst:
                out_pix[x, y] = operator(nlst)
    return out_pix

def neighbours(pix, width, height, x, y, nx=1 , ny=1 ):
    nlst = []
    for xx in range(max(y-ny, 0), min(y+ny+1 , height)):
        for yy in range(max(x-nx, 0), min(x+nx+1 , width)):
            nlst.append(pix[xx, yy])
    return nlst

def cleanPixels(img):
    # height, width = img.shape
    out1 = cv2.erode(img, np.ones((2,2), np.uint8), iterations=2)
    out2 = cv2.dilate(out1, np.ones((2,2), np.uint8), iterations=3)
    out = cv2.erode(out2, np.ones((2,2), np.uint8), iterations=1)
    return out
    
def brighten(img, phi, theta):
    maxIntensity = 255.0
    newImage0 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
    newImage = newImage0.astype(np.uint8)
    return newImage

def labelRegions(img):
    width, height = img.size