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

    new_pix = vokram.markov_chain(model, w * h)
    flood_fill(w, h, pix, iter(new_pix))
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


def tokenize(w, h, pix):
    for y in range(h):
        for x in range(w):
            yield pix[x, y]
    for x in range(w):
        for y in range(h):
            yield pix[x, y]


if __name__ == '__main__':
    main(sys.argv[1:])
