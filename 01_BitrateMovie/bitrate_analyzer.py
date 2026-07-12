from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend, safe on servers
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Packet:
    """A single demuxed packet with its presentation time and byte size."""

    time_s: float
    size_bytes: int


@dataclass(frozen=True)
class BitrateSeries:
    """Bitrate sampled over fixed-width time windows."""

    window_s: float
    times_s: np.ndarray       # window start times
    bitrate_kbps: np.ndarray  # kbit/s per window


@dataclass(frozen=True)
class BitrateStats:
    """Summary statistics computed over the windowed bitrate series."""

    duration_s: float
    packet_count: int
    total_bytes: int
    overall_kbps: float       # total size / duration
    mean_kbps: float
    median_kbps: float
    std_kbps: float
    min_kbps: float
    max_kbps: float
    p90_kbps: float
    p95_kbps: float
    p99_kbps: float
    peak_to_mean: float       # burstiness indicator
    coeff_variation: float    # std / mean, rate-change frequency proxy


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        sys.exit("ffprobe not found on PATH. Install ffmpeg first.")
    return path


def extract_packets(video_path: Path, stream: str = "v:0") -> list[Packet]:
    """Read per-packet pts_time and size for the given stream via ffprobe."""
    ffprobe = _require_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", stream,
        "-show_entries", "packet=pts_time,dts_time,size",
        "-of", "json",
        str(video_path),
    ]
    raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    payload = json.loads(raw)

    packets: list[Packet] = []
    for p in payload.get("packets", []):
        ts = p.get("pts_time") or p.get("dts_time")
        size = p.get("size")
        if ts is None or size is None:
            continue
        try:
            packets.append(Packet(time_s=float(ts), size_bytes=int(size)))
        except (TypeError, ValueError):
            continue

    packets.sort(key=lambda pk: pk.time_s)
    if not packets:
        sys.exit("No packets found. Wrong stream selector or unsupported file.")
    return packets


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def compute_series(packets: list[Packet], window_s: float) -> BitrateSeries:
    """Bin packet bytes into fixed windows and convert to kbit/s."""
    t0 = packets[0].time_s
    t_end = packets[-1].time_s
    duration = max(t_end - t0, window_s)

    n_bins = int(np.ceil(duration / window_s))
    byte_bins = np.zeros(n_bins, dtype=np.float64)

    for pk in packets:
        idx = min(int((pk.time_s - t0) / window_s), n_bins - 1)
        byte_bins[idx] += pk.size_bytes

    bitrate_kbps = (byte_bins * 8.0) / window_s / 1000.0
    times = t0 + np.arange(n_bins) * window_s
    return BitrateSeries(window_s=window_s, times_s=times, bitrate_kbps=bitrate_kbps)


def compute_stats(packets: list[Packet], series: BitrateSeries) -> BitrateStats:
    total_bytes = sum(pk.size_bytes for pk in packets)
    duration = max(packets[-1].time_s - packets[0].time_s, 1e-9)
    br = series.bitrate_kbps
    mean = float(np.mean(br))

    return BitrateStats(
        duration_s=duration,
        packet_count=len(packets),
        total_bytes=total_bytes,
        overall_kbps=(total_bytes * 8.0) / duration / 1000.0,
        mean_kbps=mean,
        median_kbps=float(np.median(br)),
        std_kbps=float(np.std(br)),
        min_kbps=float(np.min(br)),
        max_kbps=float(np.max(br)),
        p90_kbps=float(np.percentile(br, 90)),
        p95_kbps=float(np.percentile(br, 95)),
        p99_kbps=float(np.percentile(br, 99)),
        peak_to_mean=float(np.max(br) / mean) if mean else 0.0,
        coeff_variation=float(np.std(br) / mean) if mean else 0.0,
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_bitrate_over_time(series: BitrateSeries, stats: BitrateStats, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.times_s, series.bitrate_kbps, lw=1.2, color="#2563eb")
    ax.fill_between(series.times_s, series.bitrate_kbps, alpha=0.15, color="#2563eb")
    ax.axhline(stats.mean_kbps, color="#dc2626", ls="--", lw=1, label=f"mean {stats.mean_kbps:.0f} kbps")
    ax.axhline(stats.max_kbps, color="#f59e0b", ls=":", lw=1, label=f"peak {stats.max_kbps:.0f} kbps")
    ax.set_title(f"Bitrate over time (window={series.window_s}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bitrate (kbps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_distribution(series: BitrateSeries, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.bitrate_kbps, bins=40, color="#059669", alpha=0.8, edgecolor="white")
    ax.set_title("Bitrate distribution")
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("Window count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_rate_of_change(series: BitrateSeries, out: Path) -> Path:
    """Frequency/magnitude of bitrate changes: first difference between windows."""
    delta = np.diff(series.bitrate_kbps, prepend=series.bitrate_kbps[0])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.times_s, delta, lw=1, color="#7c3aed")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Bitrate change rate (delta between consecutive windows)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Delta bitrate (kbps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def print_report(video_path: Path, stats: BitrateStats, charts: list[Path]) -> None:
    print("\n" + "=" * 60)
    print(f"Bitrate analysis: {video_path.name}")
    print("=" * 60)
    print(f"Duration          : {stats.duration_s:.2f} s")
    print(f"Packets           : {stats.packet_count}")
    print(f"Total size        : {stats.total_bytes / 1e6:.2f} MB")
    print(f"Overall bitrate   : {stats.overall_kbps:.1f} kbps")
    print(f"Mean / Median     : {stats.mean_kbps:.1f} / {stats.median_kbps:.1f} kbps")
    print(f"Std dev           : {stats.std_kbps:.1f} kbps")
    print(f"Min / Max         : {stats.min_kbps:.1f} / {stats.max_kbps:.1f} kbps")
    print(f"P90 / P95 / P99   : {stats.p90_kbps:.1f} / {stats.p95_kbps:.1f} / {stats.p99_kbps:.1f} kbps")
    print(f"Peak-to-mean      : {stats.peak_to_mean:.2f}x  (burstiness)")
    print(f"Coeff. variation  : {stats.coeff_variation:.3f}  (rate-change intensity)")
    print("-" * 60)
    print("Charts written:")
    for c in charts:
        print(f"  - {c}")
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze video bitrate and plot charts.")
    ap.add_argument("video", type=Path, help="Path to the video file")
    ap.add_argument("--window", type=float, default=1.0, help="Window size in seconds (default: 1.0)")
    ap.add_argument("--stream", default="v:0", help="ffprobe stream selector (default: v:0)")
    ap.add_argument("--outdir", type=Path, default=Path("bitrate_out"), help="Output directory for charts")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        sys.exit(f"File not found: {args.video}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    packets = extract_packets(args.video, stream=args.stream)
    series = compute_series(packets, window_s=args.window)
    stats = compute_stats(packets, series)

    stem = args.video.stem
    charts = [
        plot_bitrate_over_time(series, stats, args.outdir / f"{stem}_bitrate.png"),
        plot_distribution(series, args.outdir / f"{stem}_distribution.png"),
        plot_rate_of_change(series, args.outdir / f"{stem}_change_rate.png"),
    ]
    print_report(args.video, stats, charts)


if __name__ == "__main__":
    main()