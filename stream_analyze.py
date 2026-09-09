"""Summarize remote-camera streaming telemetry from stream.log.jsonl.

The linked phone writes evt=send_side rows (its own send log, phone clock in ms
since page load, with per-frame seq / capture / send times). The server writes
evt=recv rows (every frame received, server clock) plus ws_open / ws_close and
periodic recv_summary rows.

Run:  python stream_analyze.py [stream.log.jsonl] [--code CODE]

Shows, per second, on BOTH clock domains:
  - SEND fps + capture->send gap (ms)   <- from the phone's telemetry
  - RECV fps + frame size               <- from the server's receive log
  - RECV/SEND ratio  (what fraction of sent frames the server actually got)
"""

import collections
import json
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "stream.log.jsonl"
    only_code = None
    if "--code" in sys.argv:
        only_code = sys.argv[sys.argv.index("--code") + 1]

    sends = collections.defaultdict(list)      # code -> [(seq, captureMs, sendMs)]
    recvs = collections.defaultdict(list)      # code -> [server_ts]
    recv_sizes = collections.defaultdict(list)
    total_sent_seqs = {}
    open_ws = collections.defaultdict(list)
    close_ws = collections.defaultdict(list)

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            code = rec.get("code") or rec.get("data", {}).get("code")
            if only_code and code != only_code:
                continue
            evt = rec.get("evt")
            if evt == "send_side":
                data = rec.get("data") or {}
                seq = data.get("seq", 0)
                total_sent_seqs[code] = max(total_sent_seqs.get(code, 0), seq or 0)
                for b in data.get("batch") or []:
                    if len(b) >= 3:
                        sends[code].append((b[0], b[1], b[2]))
            elif evt == "recv":
                recvs[code].append(rec.get("t", 0))
                if rec.get("size"):
                    recv_sizes[code].append(rec["size"])
            elif evt == "ws_open":
                open_ws[code].append(rec.get("t", 0))
            elif evt == "ws_close":
                close_ws[code].append(rec.get("t", 0))

    if not sends and not recvs:
        print("No telemetry rows found in", path)
        return 1

    for code in sorted(set(sends) | set(recvs)):
        print(f"\n=== {code or '(unknown)'} ===")
        s = sorted(sends.get(code, []), key=lambda r: r[0])
        r = sorted(recvs.get(code, []))
        sos = sorted(open_ws.get(code, []))
        scs = sorted(close_ws.get(code, []))

        if sos:
            print(f"connections: {len(sos)}  "
                  f"(open {sos[0]:.1f} ... close {scs[-1] if scs else 'now'})")
        if s:
            seqs = [x[0] for x in s]
            gaps = [b - a for a, b in zip([x[1] for x in s], [x[2] for x in s])]
            print(f"phone log: {len(s)} samples, last seq {seqs[-1]}")
            # per-second buckets on the PHONE clock
            s2 = sorted((x[2] / 1000.0, x[0]) for x in s)
            send_fps = _bucket_fps(s2)
            print("SEND fps (per phone-second): "
                  + _fmt(send_fps, unit="fps"))
            print("SEND capture->send gap: avg %.1f ms  max %.1f ms"
                  % (sum(gaps) / len(gaps), max(gaps)))
        if r:
            print("RECV : %d frames received (server clock)" % len(r))
            recv_fps = _bucket_fps([(t, 1) for t in r])
            print("RECV fps (per server-second): "
                  + _fmt(recv_fps, unit="fps"))
            if recv_sizes[code]:
                sz = recv_sizes[code]
                print("frame size: avg %.1f KiB  max %d KiB  (q = lower is smaller)"
                      % (sum(sz) / len(sz) / 1024, max(sz) / 1024))

        n_sent = total_sent_seqs.get(code, 0)
        n_recv = len(r)
        if n_sent and n_recv:
            ratio = n_recv / n_sent
            print(f"RATIO recv/sent = {ratio:.3f}  "
                  f"({n_recv} recv of {n_sent} sent seqs "
                  f"~ {n_sent - n_recv} dropped/lost)")
        print("TIP: compare SEND fps lines with RECV fps lines at the same "
              "wall-clock second; sustained SEND > RECV is the lag source.")
    return 0


def _bucket_fps(items):
    """items = sorted [(time_seconds, frames_at_that_time)] -> per-sec fps list"""
    start = items[0][0]
    counts = collections.defaultdict(int)
    for t, _ in items:
        counts[int(t - start)] += 1
    if len(items) == 1:
        return [items[0][1]]
    return [counts[k] for k in sorted(counts)][1:]  # drop partial first bucket


def _fmt(fps_list, unit="fps"):
    if not fps_list:
        return "n/a"
    med = sorted(fps_list)[len(fps_list) // 2]
    return (f"median {med} {unit}   min {min(fps_list)}   max {max(fps_list)}"
            f"   over {len(fps_list)} s")


if __name__ == "__main__":
    sys.exit(main())