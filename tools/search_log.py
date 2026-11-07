#!/usr/bin/env python3
"""Turn the search binary's stdout into the compact search log, plus one small status file.

The binary prints a 17-line stage table about once a second (~250 MB/day). This filter keeps
only what is worth keeping on disk:

  search/search.log   append-only, every session, tab-separated:
    <utc> range  <type> <order> <first> <end>   logical seeds [first, end) fully searched
                                                (checkpoint every --every s, and at exit;
                                                later lines for the same first supersede)
    <utc> island <type> <seed> <x> <z> <area>   one line per island: neighbouring candidates
                                                of the same island (same seed, within 2048
                                                blocks) are merged and the largest area kept
  search/status.json  overwritten once a minute: progress, rate inputs, latest stage table
  search/session.txt  overwritten per session: startup banner, warnings and errors

type is sb / lb / sub / lub (small/large biomes, bounded/unbound). Python ints keep the 64-bit
seed arithmetic exact; awk's doubles would not.
usage: search-bin ... | search_log.py <dir> <type> <order> <first> <pipelines> <batch>
"""
import json
import os
import re
import signal
import sys
import time

# Stopping the job signals the whole unit. The binary shuts down gracefully and prints its exact
# end, so keep reading until it closes the pipe rather than dying first.
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)

d, typ, order, first, pipes, batch = sys.argv[1:7]
first, pipes, batch = int(first), int(pipes), int(batch)
EVERY = float(os.environ.get("RANGE_EVERY", 300))
STATUS_EVERY = 60
LOG, STATUS = f"{d}/search.log", f"{d}/status.json"
ISLAND = re.compile(r"^(-?\d+) at (-?\d+) (-?\d+) with (\d+)$")
MERGE_BLOCKS = 2048   # candidates of one island are a few hundred blocks apart
HOLD_SECONDS = 60     # they are verified within seconds of each other (same GPU batch)
pending = []          # [first_seen, seed, x, z, area], written once held HOLD_SECONDS
ROUTINE = ("start_seed = ", "filter_", "init_seeds", "total ", "gpu_outputs.")

started = time.time()
reached = None          # highest start_seed printed (logical)
stages, block = [], []
final_end = None
log = open(LOG, "a", buffering=1)
note = open(f"{d}/session.txt", "w", buffering=1)


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def safe_end():
    # Each pipeline prints every 64 of its own batches and finishes them in order, so every
    # batch more than (64+1)*P batches behind the newest printed one is complete.
    if final_end is not None:
        return final_end
    if reached is None:
        return None
    e = reached - (64 + 1) * pipes * batch
    return e if e > first else None


written = None


def write_range():
    global written
    e = safe_end()
    if e is not None and e != written:
        written = e
        log.write(f"{utc()}\trange\t{typ}\t{order}\t{first}\t{e}\n")


def flush_islands(force=False):
    now = time.monotonic()
    for p in [p for p in pending if force or now - p[0] >= HOLD_SECONDS]:
        log.write(f"{utc()}\tisland\t{typ}\t{p[1]}\t{p[2]}\t{p[3]}\t{p[4]}\n")
        pending.remove(p)


def write_status():
    s = {"ts": time.time(), "type": typ, "order": order, "first": first,
         "started": started, "reached": reached, "end": safe_end(), "pipelines": pipes,
         "stages": stages, "done": final_end is not None}
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, STATUS)


last_range = last_status = time.monotonic()
for line in sys.stdin:
    line = line.rstrip("\n")
    if line.startswith("start_seed = "):
        v = int(line.split()[2])
        reached = v if reached is None else max(reached, v)
    elif line.startswith(("filter_", "init_seeds", "total ")):
        p = [s.strip() for s in line.split("|")]
        name, ms = p[0].split(" - ")
        io = p[2].split("->")
        block.append([name.strip(), float(ms.split()[0]), int(io[0]), int(io[1])])
        if name.strip() == "total":
            stages, block = block, []
    elif m := ISLAND.match(line):
        seed, x, z, area = (int(v) for v in m.groups())
        for p in pending:
            if p[1] == seed and abs(p[2] - x) <= MERGE_BLOCKS and abs(p[3] - z) <= MERGE_BLOCKS:
                if area > p[4]:
                    p[2:] = [x, z, area]
                break
        else:
            pending.append([time.monotonic(), seed, x, z, area])
    elif line.startswith("Start seed: "):   # graceful exit: the exact end
        m = re.match(r"Start seed: (-?\d+), Total seeds checked: (\d+)", line)
        if m:
            final_end = int(m.group(1)) + int(m.group(2))
        note.write(line + "\n")
    elif line and not line.startswith(ROUTINE):
        note.write(line + "\n")
    now = time.monotonic()
    if now - last_status >= STATUS_EVERY:
        write_status()
        last_status = now
    if pending:
        flush_islands()
    if now - last_range >= EVERY:
        write_range()
        last_range = now
flush_islands(force=True)
write_range()
write_status()
