#!/usr/bin/env python

import argparse
import collections
import itertools
import logging
import sys

from PIL import Image
import vokram


def remix(paths, ngram_size, output_size):
    imgs = map(prep_image, paths)
    out_w, out_h = output_size

    tokens_iters = []
    for img in imgs:
        w, h = img.size
        tokens_iters.append(tokenize(w, h, img.load()))

    sentinal = 0
    tokens = itertools.chain.from_iterable(tokens_iters)
    model = vokram.build_model(tokens, ngram_size, sentinal)
    start_key = (sentinal,) * ngram_size

    img_count = len(imgs)
    pixels = sum(img.size[0] * img.size[1] for img in imgs)
    logging.info('%d image(s), %d pixels', img_count, pixels)
    logging.info('Model size: %d', len(model))

    img = Image.new('RGB', output_size)
    pix = img.load()

    new_pix = vokram.markov_chain(model, start_key=start_key)
    fill(out_w, out_h, pix, new_pix)
    img = img.crop((1, 1, out_w - 1, out_h - 1))
    img.save(outfile, 'png')
    return 0


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
    arg_parser = argparse.ArgumentParser(
        prog='markovangelo',
        description='Uses Markov chains to remix images.')
    arg_parser.add_argument(
        '-n', '--ngram-size', type=int, default=4)
    arg_parser.add_argument(
        '--width', type=int, required=True,
        help='Output image width')
    arg_parser.add_argument(
        '--height', type=int, required=True,
        help='Output image height')
    arg_parser.add_argument(
        '--show', action='store_true', help='Open result in image viewer')
    arg_parser.add_argument(
        'source_file', nargs='+', help='Input image(s)')

    args = arg_parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    img = remix(args.source_file, args.ngram_size, (args.width, args.height))
    if args.show:
        img.show()
    img.save(sys.stdout, 'png')
