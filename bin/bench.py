#!/usr/bin/env python3
"""
Load generator for the /represent endpoint.

Sends N requests at a fixed concurrency and reports latency percentiles and
throughput. Use it to pick the thread split on the target host, for example:

    INFERENCE_POOL_SIZE=4 ORT_INTRA_OP_THREADS=1 ./bin/start
    ./bin/bench.py --image photo.jpg --concurrency 8 --requests 200

    INFERENCE_POOL_SIZE=2 ORT_INTRA_OP_THREADS=2 ./bin/start
    ./bin/bench.py --image photo.jpg --concurrency 8 --requests 200

A healthy run under a burst shows some 503s (back-pressure) and no 504s.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    image: bytes,
    filename: str,
    latencies: list[float],
    statuses: Counter,
) -> None:
    """Send one request and record its status and latency."""
    start = time.perf_counter()
    try:
        response = await client.post(
            url,
            files={"image_file": (filename, image, "image/jpeg")},
        )
        status = response.status_code
    except httpx.HTTPError as e:
        statuses[type(e).__name__] += 1
        return

    latencies.append((time.perf_counter() - start) * 1000)
    statuses[status] += 1


async def run(args: argparse.Namespace) -> int:
    """Run the benchmark and print the report."""
    image = Path(args.image).read_bytes()
    filename = Path(args.image).name

    latencies: list[float] = []
    statuses: Counter = Counter()
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def guarded() -> None:
            async with semaphore:
                await _one_request(
                    client, args.url, image, filename, latencies, statuses
                )

        start = time.perf_counter()
        await asyncio.gather(*(guarded() for _ in range(args.requests)))
        elapsed = time.perf_counter() - start

    ok = statuses.get(200, 0)
    print(f"requests:   {args.requests} at concurrency {args.concurrency}")
    print(f"elapsed:    {elapsed:.2f}s")
    print(f"throughput: {args.requests / elapsed:.1f} req/s "
          f"({ok / elapsed:.1f} successful req/s)")
    print(f"statuses:   {dict(sorted(statuses.items(), key=str))}")

    if latencies:
        latencies.sort()
        quantiles = statistics.quantiles(latencies, n=100, method="inclusive")
        print(f"latency ms: p50={statistics.median(latencies):.0f} "
              f"p95={quantiles[94]:.0f} "
              f"p99={quantiles[98]:.0f} "
              f"max={latencies[-1]:.0f}")

    return 0 if ok else 1


def main() -> int:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:5001/represent")
    parser.add_argument("--image", default="photo.jpg", help="Image file to send")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
