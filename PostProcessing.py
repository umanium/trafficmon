import cv2
import numpy as np

def histEqualization(im, nbr_bins=256) :
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1] ,cdf)
    return cv2.convertScaleAbs(im2.reshape(im.shape))

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
    out2 = cv2.dilate(out1, np.ones((2,2), np.uint8), iterations=4)
    out = cv2.erode(out2, np.ones((2,2), np.uint8), iterations=2)
    return out
    
def brighten(img, phi, theta):
    maxIntensity = 255.0
    newImage0 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
    newImage = newImage0.astype(np.uint8)
    return newImage

def labelRegions(img):
    width, height = img.size

def foregroundDetection(frame, bg, rect = True):
    resultant = cv2.absdiff(frame, bg)
    ret, fgMask = cv2.threshold(resultant,60,255,cv2.THRESH_BINARY)
    cleanRaw = cleanPixels(fgMask)
    clean = np.copy(cleanRaw)
    rects = []
    if rect:
        contours, hierarchy = cv2.findContours(cleanRaw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rects.append(cv2.boundingRect(contour))
    return rects, clean

def boundingBoxMask(rects, img):
    mask = np.zeros_like(img)
    for box in rects:
        x, y, w, h = box
        if x > 0 : x -= 1
        if y > 0 : y -= 1
        w += 1
        h += 1
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
    newRects = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        newRects.append(cv2.boundingRect(contour))
    return newRects