"""
get commedity information, such as catogory, name, image, ...

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/01 09:01:01
"""

from numpy import ndarray
import numpy as np
import freetype
import copy
import os
import logging
import cv2 as cv


def get_text_painters(fonts_path):
    """
    return all fonts
    """
    text_painters = {}
    for fname in os.listdir(fonts_path):
        font_name = fname.split('.')[0]
        text_painters[font_name] = PutChineseText(fonts_path + '/' + fname)
    logging.info('loaded fonts %d' % len(text_painters))
    return text_painters


def resize_zoom(img, dst_size, align='center'):
    """
    resize img by dst_size, width and height is equal proportion
    Returns:
        new img with dst_size
    """
    rate_h = float(dst_size[0]) / img.shape[0]
    rate_w = float(dst_size[1]) / img.shape[1]
    max_r = min(rate_h, rate_w)
    new_h = int(max_r * img.shape[0])
    new_w = int(max_r * img.shape[1])
    if max_r > 1:
        new_img = cv.resize(img, dsize=(new_w, new_h), \
                            fx=0, fy=0, interpolation=cv.INTER_LINEAR & cv.INTER_CUBIC)
        # cv.INTER_CUBIC
    else:
        new_img = cv.resize(img, dsize=(new_w, new_h), \
                            fx=0, fy=0, interpolation=cv.INTER_AREA)
        # cv.INTER_AREA

    # must be png now. if jpg or bmp, should add alpha channel first
    assert (img.shape[2] == 4)

    bkg = np.zeros(shape=(int(dst_size[0]), int(dst_size[1]), 4), dtype='uint8')

    if align == 'center':
        h1 = int((dst_size[0] - new_img.shape[0]) / 2)
        h2 = int(h1 + new_img.shape[0])
        w1 = int((dst_size[1] - new_img.shape[1]) / 2)
        w2 = int(w1 + new_img.shape[1])
    elif align == 'left':
        h1 = int((dst_size[0] - new_img.shape[0]) / 2)
        h2 = int(h1 + new_img.shape[0])
        w1 = 0
        w2 = int(w1 + new_img.shape[1])
    elif align == 'right':
        h1 = int((dst_size[0] - new_img.shape[0]) / 2)
        h2 = int(h1 + new_img.shape[0])
        w1 = int(dst_size[1] - new_img.shape[1])
        w2 = int(w1 + new_img.shape[1])
    elif align == 'top':
        h1 = 0
        h2 = int(h1 + new_img.shape[0])
        w1 = int(dst_size[1] - new_img.shape[1])
        w2 = int(w1 + new_img.shape[1])
    elif align == 'bottom':
        h1 = int(dst_size[0] - new_img.shape[0])
        h2 = int(h1 + new_img.shape[0])
        w1 = int(dst_size[1] - new_img.shape[1])
        w2 = int(w1 + new_img.shape[1])
    else:
        logging.fatal('unknown align %s' % align)
        assert (1 == 0)
    bkg[h1:h2, w1:w2] = new_img

    return bkg


class PutChineseText(object):
    """
    add chinese text to img
    """
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)
    
    def get_text_img(self, text, text_size, text_color, bgcolor='', image_size=(0, 0)):
        """
        return an image which contains text, with alpha channle
        Returns:
            img
        """
        if text_size < 0:
            self._face.set_char_size(100 * 64)
        else:
            self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0

        ypos = int(ascender)
        
        width, height, ypos2 = self.get_text_img_size(text, ypos)
        img = np.zeros(shape=(height, width, 4), dtype='uint8')
        self.draw_string(img, 0, ypos + ypos2, text, text_color)
        if image_size[0] > 0 and image_size[1] > 0:
            img = resize_zoom(img, image_size, 'center')

        if bgcolor != '' and bgcolor is not None:
            bgimg = np.ndarray(shape=(img.shape[0], img.shape[1], 3), dtype=np.uint8)
            bgimg[0:, 0:] = bgcolor
            mask = img[0:, 0:, 3].astype(np.float) / 255.0
            mask = mask[0:, 0:, np.newaxis]
            mask = np.concatenate([mask, mask, mask], axis=2)
            img[0:, 0:, 0: 3] = bgimg * (1.0 - mask) + img[0:, 0:, 0: 3] * mask
            img[0:, 0:, 3] = 255

        return img
    
    def get_text_img_size(self, text, ypos):
        """
        calc text img width and height
        :return:      image
        """
        prev_char = 0
        pen = freetype.Vector()
        pen.x = 0 << 6   # div 64
        pen.y = ypos << 6

        hscale = 1.0
        #matrix = freetype.Matrix(int(hscale) * 0x10000L, int(0.2 * 0x10000L),\
        #                         int(0.0 * 0x10000L), int(1.1 * 0x10000L))
        matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.0 * 0x10000),\
                                 int(0.0 * 0x10000), int(1.0 * 0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()
        
        max_y = -9999
        min_y = 9999
        width = 0
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)
            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            #max_top = max(max_top, slot.bitmap_top)
            #max_bot = max(max_bot, bitmap.rows - max_top)
            max_y = max(max_y, ypos - slot.bitmap_top + bitmap.rows)
            min_y = min(min_y, ypos - slot.bitmap_top)
            width = max(width, (pen.x >> 6) + bitmap.width)
            pen.x += slot.advance.x
            prev_char = cur_char
        return (width, - min_y + max_y, - min_y)
        
    def draw_text(self, image, pos, text, text_size, text_color):
        """
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode
        :param text_size: text size
        :param text_color:text color
        :return:          image
        """
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0

        ypos = int(ascender)

        if pos == 'center':
            width, height, ypos2 = self.get_text_img_size(text, ypos)
            # print width, height, image.shape, ypos, ypos2
            pos_x = (image.shape[1] - width) / 2
            pos_y = (image.shape[0] - height) / 2
            pos = (pos_x, pos_y + ypos2)
        
        self.draw_string(image, pos[0], pos[1] + ypos, text, text_color)

    def draw_string(self, img, x_pos, y_pos, text, color):
        """
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        """
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6   # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        #matrix = freetype.Matrix(int(hscale) * 0x10000L, int(0.2 * 0x10000L),\
        #                         int(0.0 * 0x10000L), int(1.1 * 0x10000L))
        matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.0 * 0x10000),\
                                 int(0.0 * 0x10000), int(1.0 * 0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        #image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(img, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char
            #print slot.advance.x

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        """
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        """
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer
        if img.shape[2] == 4:
            alpha = True
        else:
            alpha = False
        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row * cols + col] == 0:
                    continue
                if alpha:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]
                    img[y_pos + row][x_pos + col][3] = glyph_pixels[row * cols + col]
                else:
                    mask = glyph_pixels[row * cols + col] / 255.0
                    img[y_pos + row][x_pos + col][0] = \
                            int(img[y_pos + row][x_pos + col][0] * (1 - mask) + color[0] * mask)
                    img[y_pos + row][x_pos + col][1] = \
                            int(img[y_pos + row][x_pos + col][1] * (1 - mask) + color[1] * mask)
                    img[y_pos + row][x_pos + col][2] = \
                            int(img[y_pos + row][x_pos + col][2] * (1 - mask) + color[2] * mask)

