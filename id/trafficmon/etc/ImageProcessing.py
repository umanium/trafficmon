import numpy as np
import cv2
import math
__author__ = 'Luqman'

'''
image processing
mainly for post processing, but maybe can be used for other purposes
'''


def hist_equalization(im, nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return cv2.convertScaleAbs(im2.reshape(im.shape))


def morphological(im, operator=min, nx=5, ny=5):
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


def neighbours(pix, width, height, x, y, nx=1, ny=1):
    nlst = []
    for xx in range(max(y - ny, 0), min(y + ny + 1, height)):
        for yy in range(max(x - nx, 0), min(x + nx + 1, width)):
            nlst.append(pix[xx, yy])
    return nlst


def clean_pixels(img):
    # height, width = img.shape
    out1 = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    out2 = cv2.dilate(out1, np.ones((2, 2), np.uint8), iterations=2)
    out = cv2.erode(out2, np.ones((2, 2), np.uint8), iterations=1)
    return out


def foreground_detection(frame, bg, rect=True):
    resultant = cv2.absdiff(frame, bg)
    ret, fgmask = cv2.threshold(resultant, 65, 255, cv2.THRESH_BINARY)
    cleanraw = clean_pixels(fgmask)
    clean = np.copy(cleanraw)
    rects = []
    if rect:
        contours, hierarchy = cv2.findContours(cleanraw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rects.append(cv2.boundingRect(contour))
    return rects, clean


def foreground_process(fgraw, rect=True):
    cleanraw = clean_pixels(fgraw)
    clean = np.copy(cleanraw)
    rects = []
    if rect:
        contours, hierarchy = cv2.findContours(cleanraw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rects.append(cv2.boundingRect(contour))
    return rects, clean


def bounding_box_mask(rects, img):
    mask = np.zeros_like(img)
    for box in rects:
        x, y, w, h = box
        if (w > 3) and (h > 3):
            if x > 0:
                x -= 1
            if y > 0:
                y -= 1
            w += 1
            h += 1
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    newrects = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        newrects.append(cv2.boundingRect(contour))
    return newrects


def get_roi_from_images(rect, img):
    x, y, w, h = rect
    new_img = img[y:y + h, x:x + w]
    return new_img


def distance_two_squares(sq1, sq2):
    x1, y1, w1, h1 = sq1
    point1 = (x1+(w1/2), y1+(h1/2))
    x2, y2, w2, h2 = sq2
    point2 = (x2+(w2/2), y2+(h2/2))
    dist = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return dist


def is_new_square(square, list_of_square):
    min_dist = 100
    for existing_square in list_of_square:
        dist = distance_two_squares(square, existing_square)
        min_dist = min(dist, min_dist)
    return min_dist > 40


def hist_lines(im):
    h = np.zeros((300, 256, 3))
    if len(im.shape) != 2:
        print "hist_lines applicable only for grayscale images"
        # print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im], [0], None, [256], [0, 256])
    cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    for x, y in enumerate(hist):
        cv2.line(h, (x, 0), (x, y), (255, 255, 255))
    y = np.flipud(h)
    return y


def hist_curve(im):
    bins = np.arange(256).reshape(256, 1)
    h = np.zeros((300, 256, 3))
    if len(im.shape) == 2:
        color = [(255, 255, 255)]
    elif im.shape[2] == 3:
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins, hist)))
        cv2.polylines(h, [pts], False, col)
    y = np.flipud(h)
    return y


def get_mask_from_contour(cnt, img):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask))
    return pixelpoints


def get_mask_from_contour_2(cnt, img):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    return mask


def move_contour(cnt, direction, image_dim):
    height, width = image_dim

    def move_point(pt):
        [[x, y]] = pt
        x_dir, y_dir = direction
        x_new = x + x_dir
        if x_new < 0:
            x_new = 0
        elif x_new >= width:
            x_new = width-1
        y_new = y + y_dir
        if y_new < 0:
            y_new = 0
        elif y_new >= height:
            y_new = height-1

        return [[x_new, y_new]]

    new_cnt = np.array([move_point(pt) for pt in cnt], np.int32)
    return new_cnt


def extract_chain_code(pt1, pt2):
    # chain code:
    # 3 2 1
    # 4   0
    # 5 6 7

    [[x1, y1]] = pt1
    [[x2, y2]] = pt2
    delta_x, delta_y = x2-x1, y2-y1

    if delta_x > 0:
        h_direction = 0
    else:
        h_direction = 4

    if delta_y > 0:
        v_direction = 2
    else:
        v_direction = 6

    d_direction = (h_direction + v_direction) / 2
    if h_direction == 0 and v_direction == 6:
        d_direction = 7

    absolute_delta_x = np.abs(delta_x)
    absolute_delta_y = np.abs(delta_y)

    non_diagonal = absolute_delta_x - absolute_delta_y
    non_diagonal_h = non_diagonal > 0
    if non_diagonal_h:
        nd_direction = h_direction
    else:
        nd_direction = v_direction
    non_diagonal_count = np.abs(non_diagonal)
    diagonal_count = min(absolute_delta_x, absolute_delta_y)

    diagonal_chain_code = [d_direction, diagonal_count]
    non_diagonal_chain_code = [nd_direction, non_diagonal_count]

    return [diagonal_chain_code, non_diagonal_chain_code]