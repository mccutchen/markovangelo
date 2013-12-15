import collections

import Image
import vokram


def main():
    img = Image.open('monalisa.gif')
    w, h = img.size
    pix = img.load()
    tokens = tokenize(w, h, pix)
    model = vokram.build_model(tokens, 10)

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
    main()
