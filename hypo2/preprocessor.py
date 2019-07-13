from hypo2.base.basef import BaseHIObj
from scipy import misc
from scipy import ndimage
import numpy as np
import cv2 as cv
import hypo2.base.exceptions as exc


class Preprocessor(BaseHIObj):
    def __init__(self, config):
        self.config = config.copy()
        self.norm_methods = {
            'blue': self.normalize_color_blue,
            'otsu': self.normalize_color_otsu,
            'otsu_mask': self.normalize_color_otsu_mask,
            'adaptive_mean': self.normalize_color_adaptive_mean,
            'gaussian': self.normalize_color_gaussian
        }

    def open(self, path):
        return misc.imread(path)
    
    def norm(self, img, method='otsu_mask'):
        img = self.auto_contrast(img)
        img = self.norm_methods[method](img)
        if self.config.SHEET_ANGLE == "adaptive":
            img = self.rotate(img)
        elif type(self.config.SHEET_ANGLE) in [int, float]:
            pass
        else:
            exc.raiseExcep("config.SHEET_ANGLE set incorrectly!")
        img = self.inverse(img)
        img = self.auto_contrast(img)
        return img
    
    def open_norm(self, path, method='otsu_mask'):
        return self.norm(self.open(path), method=method)

    def open_norm_segm(self, path):
        norm_image = self.open_norm(path)
        words = self.segment_words(norm_image)
        return words

    def normalize_color_blue(self, img):
        img = img.copy()
        r, g, b = img.T

        mask = ((b / 1.1 > g) & (b / 1.1 > r)).T
        img[mask] = (255, 255, 255)
        img[~mask] = np.array((0, 0, 0))
        return img

    def otsu_thresh(self, img):
        morph = img.copy()
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
        morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

        # take morphological gradient
        gradient_image = cv.morphologyEx(morph, cv.MORPH_GRADIENT, kernel)

        # split the gradient image into channels
        image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

        channel_height, channel_width, _ = image_channels[0].shape

        # apply Otsu threshold to each channel
        for i in range(0, 3):
            _, image_channels[i] = cv.threshold(~image_channels[i], 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
            image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

        # merge the channels
        image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
        
        img = cv.cvtColor(image_channels, cv.COLOR_BGR2GRAY)
        return img
    
    def normalize_color_otsu(self, img):
        img = self.otsu_thresh(img)
        img = 255 - np.stack((img,)*3, axis=-1)
        return img
    
    def normalize_color_otsu_mask(self, img):
        img = img.copy()
        mask = np.where(self.otsu_thresh(img) > 127, True, False)
        mask = ~ndimage.binary_fill_holes(~mask)
        img[mask] = (255, 255, 255)
        img = 255 - img
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = np.stack((img,)*3, axis=-1)
        img = self.auto_contrast(img)
        return img
    
    def normalize_color_adaptive_mean(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(img, 5)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        img = 255 - np.stack((img,)*3, axis=-1)
        return img
    
    def normalize_color_gaussian(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(img, 5)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        img = 255 - np.stack((img,)*3, axis=-1)
        return img
    
    def auto_contrast(self, img):
        with np.errstate(divide='ignore', invalid='ignore'):
            return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype('uint8')
    
    def inverse(self, img):
        return 255 - img
    
    def rotate(self, img):
        maxi = 0
        max_lines = -1
        i = -3
        while i <= 3:
            cnt = self.count_lines(255-misc.imrotate(img, i))
            if cnt > max_lines:
                max_lines = cnt
                maxi = i
            i += 0.25
        return misc.imrotate(img, maxi)
            
    def count_lines(self, img):
        cnt = len(self.get_ray_lines(np.array(img), self.config.VERT_RAY_THRESHOLD)) // 2
        return cnt

    def __const_get_ray_lines(self, img, thr):
        lines = [0]
        turn = 0
        sums = np.sum(255 - np.sum(img, axis=2) / 3, axis=1) / 255
        for s in sums:
            if s > thr:
                if turn == 0:
                    turn = 1
                    lines.append(1)
                else:
                    lines[-1] += 1
            else:
                if turn == 1:
                    turn = 0
                    lines.append(1)
                else:
                    lines[-1] += 1
        return lines

    def __search(self, f, begin, end, cache):
        if end - begin <= 3:
            l = list(range(begin, end))
            for i in l:
                if i not in cache:
                    cache[i] = f(i)
            return np.argmax(np.array((map(lambda x: cache[x], range(begin, end))))) + begin
        low = (begin * 2 + end) // 3
        high = (begin + end * 2) // 3
        if low not in cache:
            cache[low] = f(low)
        if high not in cache:
            cache[high] = f(high)
        if cache[low] > cache[high]:
            return self.__search(f, begin, high - 1, cache)
        else:
            return self.__search(f, low + 1, end, cache)

    def search(self, f, begin, end):
        cache = {}
        return self.__search(f, begin, end, cache)

    def get_ray_lines(self, img, thr):
        if type(thr) in [float, int]:
            return self.__const_get_ray_lines(img, thr)
        elif type(thr) == str and thr == "adaptive":
            f = lambda x: self.__const_get_ray_lines(img, x)
            '''
            lns = []
            bs = -1
            for i in range(5, 70, 5):
                lns_new = self.__const_get_ray_lines(img, i)
                if len(lns_new) > len(lns):
                    lns = lns_new
                    bs = i
            return lns
            '''
            return self.__const_get_ray_lines(img, self.search(f, 2, 100))
        else:
            exc.raiseExcep("Not valid threshold! thr =", str(thr), "but is required to be number or string ('adaptive')")
    
    def get_rbounds(self, lines, thrmin, thrmax):
        rrs = []
        s = 0
        for i in lines:
            s += i
            rrs.append(s)
        bounds = []
        for i in range(len(rrs) // 2 - 1):
            if abs(rrs[i * 2] - rrs[i * 2 + 1]) > thrmin and abs(rrs[i * 2] - rrs[i * 2 + 1]) < thrmax:
                bounds.append((rrs[i * 2], rrs[i * 2 + 1]))
        return bounds
    
    def get_chunks(self, img, rbounds):
        chunks = []
        for r in rbounds:
            chunks.append(img[r[0]:r[1]])
        return chunks
    
    def add_padding(self, res, pattern):
        before_0 = -(res.shape[1] - pattern[0]) // 2
        after_0 = -(res.shape[1] - pattern[0]) // 2 + (res.shape[1] - pattern[0]) % 2
        before_1 = -(res.shape[0] - pattern[1]) // 2
        after_1 = -(res.shape[0] - pattern[1]) // 2 + (res.shape[0] - pattern[1]) % 2
        return np.pad(res, [(before_1, after_1), (before_0, after_0), (0, 0)], mode="constant", constant_values=255)
    
    def segment_words(self, img):
        lines = self.get_ray_lines(img, self.config.VERT_RAY_THRESHOLD)
        rbounds = self.get_rbounds(lines, self.config.VERT_RAY_CHUNKMINSIZE, self.config.VERT_RAY_CHUNKMAXSIZE)
        self.chunks = self.get_chunks(img, rbounds)
        res = []
        for chunk in self.chunks:
            transposed_chunk = chunk.transpose((1, 0, 2))
            wlines = self.get_ray_lines(transposed_chunk, self.config.HORIZ_RAY_THRESHOLD)
            wbounds = self.get_rbounds(wlines, self.config.HORIZ_RAY_CHUNKMINSIZE, self.config.HORIZ_RAY_CHUNKMAXSIZE)
            words = self.get_chunks(transposed_chunk, wbounds)
            res.extend(words)
        for i in range(len(res)):
            res[i] = self.add_padding(res[i].transpose((1, 0, 2)), self.config.FINAL_SIZE)
            res[i] = self.auto_contrast(res[i])
            res[i] = (1 - res[i] / 255) - 1/2
        return res
