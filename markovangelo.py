#!/usr/bin/env python

import collections
import itertools
import logging
import sys

import Image
import vokram


def main(paths):
    imgs = map(prep_image, paths)

    tokens_iters = []
    for img in imgs:
        w, h = img.size
        pix = img.load()
        tokens_iters.append(tokenize(w, h, pix))

    ngram_size = 8
    sentinal = 0
    tokens = itertools.chain.from_iterable(tokens_iters)
    model = vokram.build_model(tokens, ngram_size, sentinal)
    start_key = (sentinal,) * ngram_size

    img_count = len(imgs)
    pixels = sum(img.size[0] * img.size[1] for img in imgs)
    logging.info('%d image(s), %d pixels', img_count, pixels)
    logging.info('Model size: %d', len(model))

    w = 500
    h = 500
    img = Image.new('RGB', (w, h))
    pix = img.load()

    new_pix = vokram.markov_chain(model, start_key=start_key)
    fill(w, h, pix, new_pix)
    img = img.crop((1, 1, w - 1, h - 1))
    img.save(sys.stdout, 'png')


def prep_image(path):
    return Image.open(path).quantize(colors=256).convert('RGB')


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
    for y in range(1, h - 1, 2):
        for x in range(1, w - 1, 2):
            pix[x, y] = next(new_pix)
            for nx, ny in neighbors(x, y):
                pix[nx, ny] = next(new_pix)


def tokenize(w, h, pix):
    """We tokenize an image such that there is a token for each pixel
    and each of its neighboring pixels, so that each neighbor is
    equally likely to occur after any given pixel.

    (And we ignore the outermost pixels for simplicity's sake.)
    """
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for nx, ny in neighbors(x, y):
                yield pix[x, y]
                yield pix[nx, ny]


def neighbors(x, y):
    return [
        (x - 1, y),
        (x - 1, y - 1),
        (x, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
        (x, y + 1),
    ]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main(sys.argv[1:])
