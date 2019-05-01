import numpy as np
from hypo2.basef import BaseHIObj


class WordSegmentator(BaseHIObj):
    def __init__(self, config):
        self.config = config

    def get_ray_lines(self, im, thr):
        lines = [0]
        turn = 0
        for i in range(im.shape[0]):
            s = sum(255 - sum(g) / 3 for g in im[i]) / 255
            if s > thr:
                if turn == 0:
                    turn = 1 - turn
                    lines.append(1)
                else:
                    lines[-1] += 1
            else:
                if turn == 1:
                    turn = 1 - turn
                    lines.append(1)
                else:
                    lines[-1] += 1
        return lines

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

    def get_chunks(self, im, rbounds):
        chunks = []
        for r in rbounds:
            chunks.append(im[r[0]:r[1]])
        return chunks

    def add_padding(self, res, pattern):
        before_0 = -(res.shape[1] - pattern[0]) // 2
        after_0 = -(res.shape[1] - pattern[0]) // 2 + (res.shape[1] - pattern[0]) % 2
        before_1 = -(res.shape[0] - pattern[1]) // 2
        after_1 = -(res.shape[0] - pattern[1]) // 2 + (res.shape[0] - pattern[1]) % 2
        return np.pad(res, [(before_1, after_1), (before_0, after_0), (0, 0)], mode="constant", constant_values=255)

    def segment_words(self, im):
        lines = self.get_ray_lines(im, self.config.VERT_RAY_THRESHOLD)
        rbounds = self.get_rbounds(lines, self.config.VERT_RAY_CHUNKMINSIZE, self.config.VERT_RAY_CHUNKMAXSIZE)
        self.chunks = self.get_chunks(im, rbounds)
        res = []
        for chunk in self.chunks:
            transposed_chunk = chunk.transpose((1, 0, 2))
            wlines = self.get_ray_lines(transposed_chunk, self.config.HORIZ_RAY_THRESHOLD)
            wbounds = self.get_rbounds(wlines, self.config.HORIZ_RAY_CHUNKMINSIZE, self.config.HORIZ_RAY_CHUNKMAXSIZE)
            words = self.get_chunks(transposed_chunk, wbounds)
            res.extend(words)
        for i in range(len(res)):
            res[i] = self.add_padding(res[i].transpose((1, 0, 2)), self.config.FINAL_SIZE)
        return res
