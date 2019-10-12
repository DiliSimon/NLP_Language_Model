import util
import argparse
from collections import defaultdict


def load_words(filename):
    vecs = defaultdict()
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            words = l.split()
            key = words[0]
            vec = words[1:]
            for idx, v in enumerate(vec):
                vec[idx] = float(v)
            vecs[key] = vec
    return dict(vecs)


def most_similar_words(vec, target_vec, target = None):
    sim = defaultdict()
    for w in vec.keys():
        if w == target:
            continue
        sim[util.cosine_sim(vec[w], target_vec)] = w
    i = 0
    for v in sorted(sim.keys(), reverse=True):
        if i == 9:
            print(sim[v])
        else:
            print(sim[v], end=' ')
        i += 1
        if i > 9:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument("--minus", type=str)
    parser.add_argument("--plus", type=str)
    args = parser.parse_args()
    vecs = load_words(args.filename)
    if args.minus is None:
        most_similar_words(vecs, vecs[args.target], args.target)
    else:
        target_vec = util.combine_vecs(vecs, args.target, args.minus, args.plus)
        most_similar_words(vecs, target_vec)
