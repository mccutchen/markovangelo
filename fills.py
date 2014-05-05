import collections
import itertools
import random

import utils


def simple_fill(w, h, target_pix, pix_stream):
    for y in range(0, h):
        for x in range(0, w):
            target_pix[x, y] = next(pix_stream)


def less_simple_fill(w, h, target_pix, pix_stream):
    for y in range(2, h - 2, 2):
        for x in range(2, w - 2, 2):
            target_pix[x, y] = next(pix_stream)
            for nx, ny in utils.neighbors(x, y):
                target_pix[nx, ny] = next(pix_stream)


def stride_fill(w, h, target_pix, pix_stream):
    stride = int(h * .1)
    for row_y in range(0, h, stride):
        for x in range(0, w):
            for y in range(row_y, min(row_y + stride, h)):
                target_pix[x, y] = next(pix_stream)


def random_stride_fill(w, h, target_pix, pix_stream):
    base_stride = int(h * .05)
    stride_range = int(base_stride * .33)
    row_y = 0
    while row_y < h:
        stride = base_stride + random.randint(-stride_range, stride_range)
        for x in range(0, w):
            for y in range(row_y, min(row_y + stride, h)):
                target_pix[x, y] = next(pix_stream)
        row_y += stride


def flood_fill(w, h, target_pix, pix_stream):
    visited = set()
    q = collections.deque()
    q.append((0, 0))

    while q:
        x, y = q.popleft()
        if not 0 <= x < w or not 0 <= y < h or (x, y) in visited:
            continue

        visited.add((x, y))
        target_pix[x, y] = next(pix_stream)

        coords = [
            (x + 1, y),
            (x, y + 1)
        ]
        q.extend(coords)


def patchwork_fill(w, h, target_pix, pix_stream):
    # This fill breaks the images into square patches and fills each patch
    # individually.
    patch_size = int(max(w, h) * 0.025)
    patch_x_range = xrange(0, w, patch_size)
    patch_y_range = xrange(0, h, patch_size)
    patch_coords = list(itertools.product(patch_y_range, patch_x_range))
    patch_coords = sorted(patch_coords, reverse=True)

    for patch_y, patch_x in patch_coords:
        x_range = xrange(patch_x, min(patch_x + patch_size, w))
        y_range = xrange(patch_y, min(patch_y + patch_size, h))
        patch_pixels = list(itertools.product(x_range, y_range))
        patch_pixels = sorted(patch_pixels, key=lambda (x, y): (y * x))
        for x, y in patch_pixels:
            target_pix[x, y] = next(pix_stream)
