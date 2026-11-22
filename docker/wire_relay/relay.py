"""Plain TCP relay with per-connection logging. No protocol parsing.

It listens on one port and forwards every connection to one upstream address. For each connection
it writes, into --out-dir,

  <name>.connections.jsonl       one open record and one close record per connection
  <name>.<conn>.c2s.bin          raw bytes from the client side (SuperNode) to the upstream (SuperLink)
  <name>.<conn>.s2c.bin          raw bytes in the other direction

The close record holds byte counts, first and last timestamps, a rolling SHA-256 of the whole
direction and one entry per forwarded chunk: [seconds since open, bytes, chunk sha256, rolling sha256].
Dumps are size capped. A capped direction is flagged as truncated and keeps being forwarded.
Standard library only.
"""
import argparse
import asyncio
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

CHUNK = 65536


class DirectionLog:
    def __init__(self, path, cap_bytes, max_events, budget):
        self.path, self.cap, self.max_events, self.budget = Path(path), cap_bytes, max_events, budget
        self.fh = self.path.open("wb")
        self.rolling = hashlib.sha256()
        self.total = self.dumped = 0
        self.events, self.events_dropped = [], 0
        self.truncated = False
        self.first = self.last = None

    def add(self, chunk, t0):
        now = time.time()
        self.first = self.first if self.first is not None else now
        self.last = now
        self.total += len(chunk)
        self.rolling.update(chunk)
        room = min(self.cap - self.dumped, self.budget["left"])
        if room >= len(chunk):
            self.fh.write(chunk)
            self.fh.flush()
            self.dumped += len(chunk)
            self.budget["left"] -= len(chunk)
        else:
            self.truncated = True
            if room > 0:
                self.fh.write(chunk[:room])
                self.fh.flush()
                self.dumped += room
                self.budget["left"] -= room
        if len(self.events) < self.max_events:
            self.events.append([round(now - t0, 6), len(chunk), hashlib.sha256(chunk).hexdigest(),
                                self.rolling.hexdigest()])
        else:
            self.events_dropped += 1

    def summary(self):
        self.fh.close()
        return {"bytes": self.total, "dump_file": self.path.name, "dump_bytes": self.dumped,
                "truncated": self.truncated, "sha256": self.rolling.hexdigest(), "first_t": self.first,
                "last_t": self.last, "chunks": self.events, "events_dropped": self.events_dropped}


class Relay:
    def __init__(self, name, upstream_host, upstream_port, out_dir, dump_cap, total_cap, max_events):
        self.name, self.up = name, (upstream_host, upstream_port)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dump_cap, self.max_events = dump_cap, max_events
        self.budget = {"left": total_cap}
        self.n = 0
        self.log = (self.out / f"{name}.connections.jsonl").open("a", buffering=1)
        self.tasks = set()

    def emit(self, record):
        self.log.write(json.dumps(record, sort_keys=True) + "\n")
        self.log.flush()

    async def pump(self, reader, writer, dlog, t0):
        """Copy one direction. End of input is passed on as a half close so replies can still arrive."""
        try:
            while True:
                chunk = await reader.read(CHUNK)
                if not chunk:
                    break
                dlog.add(chunk, t0)
                writer.write(chunk)
                await writer.drain()
            if writer.can_write_eof():
                writer.write_eof()
        except ConnectionError:
            try:
                writer.close()
            except Exception:
                pass

    async def handle(self, client_reader, client_writer):
        self.n += 1
        conn = f"c{self.n:04d}"
        t0 = time.time()
        peer = client_writer.get_extra_info("peername")
        self.emit({"event": "open", "relay": self.name, "conn": conn, "t": t0,
                   "peer": f"{peer[0]}:{peer[1]}" if peer else None, "upstream": f"{self.up[0]}:{self.up[1]}"})
        c2s = DirectionLog(self.out / f"{self.name}.{conn}.c2s.bin", self.dump_cap, self.max_events, self.budget)
        s2c = DirectionLog(self.out / f"{self.name}.{conn}.s2c.bin", self.dump_cap, self.max_events, self.budget)
        error = None
        try:
            up_reader, up_writer = await asyncio.open_connection(*self.up)
            try:
                await asyncio.gather(self.pump(client_reader, up_writer, c2s, t0),
                                     self.pump(up_reader, client_writer, s2c, t0))
            finally:
                up_writer.close()
        except asyncio.CancelledError:
            error = "cancelled"
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                client_writer.close()
            except Exception:
                pass
            self.emit({"event": "close", "relay": self.name, "conn": conn, "t": time.time(), "error": error,
                       "c2s": c2s.summary(), "s2c": s2c.summary()})

    async def serve(self, host, port):
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass

        async def tracked(r, w):
            task = asyncio.current_task()
            self.tasks.add(task)
            try:
                await self.handle(r, w)
            finally:
                self.tasks.discard(task)

        server = await asyncio.start_server(tracked, host, port)
        sockets = server.sockets or []
        self.port = sockets[0].getsockname()[1] if sockets else port
        print(f"relay {self.name} listening on {host}:{self.port} -> {self.up[0]}:{self.up[1]}", flush=True)
        await stop.wait()
        server.close()
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.log.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--listen-host", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=9092)
    p.add_argument("--upstream", required=True, help="host:port")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dump-cap-mib", type=float, default=64.0, help="per connection and direction")
    p.add_argument("--total-cap-mib", type=float, default=256.0)
    p.add_argument("--max-chunk-events", type=int, default=100000)
    a = p.parse_args(argv)
    host, _, port = a.upstream.rpartition(":")
    relay = Relay(a.name, host, int(port), a.out_dir, int(a.dump_cap_mib * 2 ** 20),
                  int(a.total_cap_mib * 2 ** 20), a.max_chunk_events)
    asyncio.run(relay.serve(a.listen_host, a.listen_port))


if __name__ == "__main__":
    sys.exit(main())
