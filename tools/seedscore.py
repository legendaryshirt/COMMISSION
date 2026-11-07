#!/usr/bin/env python3
"""Exact replica of filter_seeds (gpu.cu) for given seeds: prints the 6 octave yo terms and the
running score, to see how far known islands sit below the 0.038 threshold.
usage: seedscore.py [--lb] < lines with a seed in the first or second column"""
import sys
import numpy as np
M = (1 << 64) - 1
SILVER, GOLDEN = 0x6a09e667f3bcc909, 0x9e3779b97f4a7c15
HASH_C = (0x83886c9d0ae3a662, 0xafa638a61b42e8ad)
HASH_CL = (0x9a3f51a113fce8dc, 0xee2dbd157e5dcdad)
OCT = [(0xb198de63a8012672, 0x7b84cad43ef7b5a8), (0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8),
       (0x36d326eed40efeb2, 0x5be9ce18223c636a), (0x082fe255f8be6631, 0x4e96119e22dedc81),
       (0x0ef68ec68504005e, 0x48b6bf93a2789640), (0xf11268128982754f, 0x257a1d670430b0aa),
       (0xe51c98ce7d1de664, 0x5f9478a733040c45), (0x6d7b49e7e429850a, 0x2e3063c622a24777),
       (0xbd90d5377ba1b762, 0xc07317d419a7548d), (0x53d39c6752dac858, 0xbcd1c5a80ab65b3e),
       (0xb4a24d7a84e7677b, 0x023ff9668e89b5c4), (0xdffa22b534c5f608, 0xb9b67517d3665ca9),
       (0xd50708086cef4d7c, 0x6e1651ecc7f43309)]
def rol(a, b): return ((a << b) | (a >> (64 - b))) & M
def mix64(a):
    a = ((a ^ (a >> 30)) * 0xbf58476d1ce4e5b9) & M
    a = ((a ^ (a >> 27)) * 0x94d049bb133111eb) & M
    return a ^ (a >> 31)
def step(l, h):  # one nextInternal: returns (r, l', h')
    r = (rol((l + h) & M, 17) + l) & M
    h ^= l
    return r, rol(l, 49) ^ h ^ ((h << 21) & M), rol(h, 28)
def yo(fork, octhash):
    _, l, h = step(*fork)                 # noise_yo_fork: skip r
    _, hl, hh = step(*octhash)            # octave_yo_fork_hash
    r, _, _ = step(l ^ hl, h ^ hh)
    return np.float32((r >> 32) & 0xFFFFFF) * np.float32(5.9604645e-8)
def terms(seed, lb=False):
    s = (seed & M) ^ SILVER
    l, h = mix64(s), mix64((s + GOLDEN) & M)
    r1, l, h = step(l, h); r2, _, _ = step(l, h)
    hc = HASH_CL if lb else HASH_C
    nl, nh = r1 ^ hc[0], r2 ^ hc[1]
    a0, nl, nh = step(nl, nh); a1, nl, nh = step(nl, nh)
    b0, nl, nh = step(nl, nh); b1, nl, nh = step(nl, nh)
    first = -11 if lb else -9
    o = [OCT[first + 12 + i] for i in range(3)]
    w = [np.float32(v) for v in (0.35, 0.35, 0.11, 0.11, 0.04, 0.04)]
    c = [yo((a0, a1), o[0]), yo((b0, b1), o[0]), yo((a0, a1), o[1]), yo((b0, b1), o[1]), yo((a0, a1), o[2]), yo((b0, b1), o[2])]
    score = np.float32(0); run = []
    for wi, ci in zip(w, c):
        score = np.float32(score + wi * np.float32(abs(ci - np.float32(0.5))))
        run.append(float(score))
    return run
if __name__ == "__main__":
    lb = "--lb" in sys.argv
    for line in sys.stdin:
        p = line.split()
        if not p or p[0].startswith("#"): continue
        seed = int(p[1] if len(p) > 4 and p[0].isdigit() and len(p[0]) < 4 else p[0])
        run = terms(seed, lb)
        print(seed, " ".join(f"{v:.5f}" for v in run), *p[2:])
