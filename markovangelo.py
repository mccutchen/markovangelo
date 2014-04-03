#!/usr/bin/env python

import collections
import itertools
import sys

import Image
import vokram


def main(paths):
    imgs = map(Image.open, paths)

    tokens_iters = []
    for img in imgs:
        w, h = img.size
        pix = img.load()
        tokens_iters.append(tokenize(w, h, pix))

    tokens = itertools.chain.from_iterable(tokens_iters)
    model = vokram.build_model(tokens, 10)

    img_count = len(imgs)
    pixels = sum(img.size[0] * img.size[1] for img in imgs)
    print('{} image(s), {} pixels'.format(img_count, pixels))
    print('Model size: {}'.format(len(model)))

    new_pix = vokram.markov_chain(model, w * h * 2)
    fill(w, h, pix, iter(new_pix))
    img.show()


def flood_fill(w, h, pix, new_pix):
    visited = set()
    q = collections.deque()
    q.append((0, 0))

    while q:
        x, y = q.popleft()
        if not 0 <= x < w or not 0 <= y < h or (x, y) in visited:
            continue

        visited.add((x, y))
        pix[x, y] = next(new_pix)

        coords = [
            (x + 1, y),
            (x, y + 1)
        ]
        q.extend(coords)


def fill(w, h, pix, new_pix):
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            pix[x, y] = next(new_pix)
            if 0 < x < w - 1 and 0 < y < h - 1:
                pix[x - 1, y] = next(new_pix)
                pix[x - 1, y - 1] = next(new_pix)
                pix[x, y - 1] = next(new_pix)
                pix[x + 1, y] = next(new_pix)
                pix[x + 1, y + 1] = next(new_pix)
                pix[x, y + 1] = next(new_pix)


def tokenize(w, h, pix):
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            yield pix[x, y]
            if 0 < x < w - 1 and 0 < y < h - 1:
                yield pix[x - 1, y]
                yield pix[x - 1, y - 1]
                yield pix[x, y - 1]
                yield pix[x + 1, y]
                yield pix[x + 1, y + 1]
                yield pix[x, y + 1]


if __name__ == '__main__':
    main(sys.argv[1:])
