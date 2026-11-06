#!/usr/bin/env python3
"""
camera.py - Test an Intel RealSense D455 and measure how much data a single camera gives.

What it does:
  1. Lists device info (name, serial, firmware, USB type).
  2. Lists every sensor and every stream profile it supports (resolution / format / fps).
  3. Prints calibration (intrinsics for each stream, depth scale, depth->color extrinsics).
  4. Streams depth + color + infrared left/right + IMU (accel/gyro) at the same time
     for N seconds and reports, per stream:
        - real FPS vs requested FPS
        - dropped frames (from frame numbers)
        - bytes per frame and bandwidth (MB/s)
     plus a total bandwidth and depth-quality stats (valid pixels, min/max/median range).
  5. Optionally saves one sample of each stream to ./camera_samples/.

Install:
  pip install pyrealsense2 numpy          # Linux / Windows
  pip install pyrealsense2-macosx numpy   # macOS (community wheel)
  pip install opencv-python               # optional, only for --save / --show

  Note: on macOS librealsense often needs root to claim the USB device:
      sudo python3 camera.py

Usage:
  python3 camera.py                       # default: 848x480 depth/IR @30, 1280x720 color @30, 10 s
  python3 camera.py --seconds 20 --save
  python3 camera.py --depth 1280x720 --color 1280x800 --fps 30
  python3 camera.py --fps 90 --depth 848x480 --color 848x480 --fps-color 60
  python3 camera.py --list-only           # only print capabilities, no streaming
  python3 camera.py --show                # live preview window (needs opencv)

  Fallback (macOS, when librealsense cannot open the camera):
  python3 camera.py --opencv              # D455 RGB as a normal webcam, OpenCV index 0
  python3 camera.py --opencv 2 --color 1280x720 --seconds 20 --save --show
  -> colour only: depth / infrared / IMU are NOT available through this path.
"""

import argparse
import collections
import os
import sys
import time

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # only required outside --opencv mode (checked in main)

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed: pip install numpy")

try:
    import cv2  # optional
except ImportError:
    cv2 = None


# --------------------------------------------------------------------------- helpers

def parse_res(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:7.2f} {unit}"
        n /= 1024
    return f"{n:7.2f} TB"


def safe_info(dev, key):
    try:
        return dev.get_info(key) if dev.supports(key) else "n/a"
    except Exception:
        return "n/a"


def stream_key(profile):
    """Unique name for a stream, e.g. 'infrared_1'."""
    name = str(profile.stream_type()).split(".")[-1]
    idx = profile.stream_index()
    return f"{name}_{idx}" if idx else name


# --------------------------------------------------------------------------- 1-3. capabilities

def print_device_info(dev):
    print("=" * 78)
    print("DEVICE")
    print("=" * 78)
    for label, key in [
        ("Name", rs.camera_info.name),
        ("Serial", rs.camera_info.serial_number),
        ("Firmware", rs.camera_info.firmware_version),
        ("Recommended FW", rs.camera_info.recommended_firmware_version),
        ("Product line", rs.camera_info.product_line),
        ("USB type", rs.camera_info.usb_type_descriptor),
        ("Physical port", rs.camera_info.physical_port),
    ]:
        val = safe_info(dev, key)
        if key == rs.camera_info.physical_port and len(val) > 60:
            val = val[:57] + "..."
        print(f"  {label:<16}: {val}")

    usb = safe_info(dev, rs.camera_info.usb_type_descriptor)
    if usb.startswith("2"):
        print("  !! Camera is on USB 2.x - bandwidth and resolutions will be limited."
              " Use a USB 3 port/cable.")


def print_sensors_and_profiles(dev):
    print()
    print("=" * 78)
    print("SENSORS AND SUPPORTED STREAM PROFILES")
    print("=" * 78)
    for sensor in dev.query_sensors():
        name = safe_info(sensor, rs.camera_info.name)
        profiles = sensor.get_stream_profiles()
        print(f"\n[{name}]  ({len(profiles)} profiles)")

        # group: stream -> format -> resolution -> [fps]
        groups = collections.OrderedDict()
        for p in profiles:
            key = (stream_key(p), str(p.format()).split(".")[-1])
            if p.is_video_stream_profile():
                vp = p.as_video_stream_profile()
                res = f"{vp.width()}x{vp.height()}"
            else:
                res = "-"
            groups.setdefault(key, collections.OrderedDict()).setdefault(res, set()).add(p.fps())

        for (stream, fmt), resolutions in groups.items():
            print(f"  {stream:<12} {fmt:<8}")
            for res, fps in resolutions.items():
                fps_txt = ",".join(str(f) for f in sorted(fps))
                print(f"      {res:<11} fps: {fps_txt}")

        # sensor options (exposure, laser power, ...)
        opts = []
        for opt in sensor.get_supported_options():
            try:
                opts.append(f"{str(opt).split('.')[-1]}={sensor.get_option(opt):g}")
            except Exception:
                pass
        if opts:
            print(f"  options ({len(opts)}): " + ", ".join(opts))


def print_calibration(profile):
    print()
    print("=" * 78)
    print("CALIBRATION")
    print("=" * 78)
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    print(f"  Depth scale : {depth_sensor.get_depth_scale():.6f} m / unit")
    if hasattr(rs.option, "stereo_baseline") and depth_sensor.supports(rs.option.stereo_baseline):
        print(f"  Baseline    : {depth_sensor.get_option(rs.option.stereo_baseline):.2f} mm")

    streams = {stream_key(s): s for s in profile.get_streams()}
    for key, s in streams.items():
        if s.is_video_stream_profile():
            i = s.as_video_stream_profile().get_intrinsics()
            print(f"  {key:<12} {i.width}x{i.height}  fx={i.fx:.2f} fy={i.fy:.2f} "
                  f"ppx={i.ppx:.2f} ppy={i.ppy:.2f}  model={str(i.model).split('.')[-1]}")
        elif s.is_motion_stream_profile():
            try:
                mi = s.as_motion_stream_profile().get_motion_intrinsics()
                print(f"  {key:<12} bias={['%.4f' % b for b in mi.bias_variances]} "
                      f"noise={['%.4f' % n for n in mi.noise_variances]}")
            except Exception:
                print(f"  {key:<12} (no motion intrinsics)")

    if "depth" in streams and "color" in streams:
        e = streams["depth"].get_extrinsics_to(streams["color"])
        t = [f"{v * 1000:.2f}" for v in e.translation]
        print(f"  Depth->Color translation (mm): {t}")


def label(img, text, sub=None):
    """Draw a title (and optional second line) in the top-left corner of a view."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 64 if sub else 36), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if sub:
        cv2.putText(img, sub, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def two_view_window(title, left, right, max_width=1600):
    """Show two views side by side in one window, scaled to fit the screen."""
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    both = np.hstack([left, right])
    if both.shape[1] > max_width:
        f = max_width / both.shape[1]
        both = cv2.resize(both, (max_width, int(both.shape[0] * f)))
    cv2.imshow(title, both)
    return cv2.waitKey(1) & 0xFF


def depth_view(depth_frame, depth_scale, max_m=6.0):
    """Colourised depth (near = red, far = blue) with the centre distance printed."""
    d = np.asanyarray(depth_frame.get_data())
    m = d * depth_scale
    norm = np.clip(255 - m / max_m * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vis[d == 0] = 0  # no depth -> black
    h, w = d.shape
    centre = depth_frame.get_distance(w // 2, h // 2)
    cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    valid = 100.0 * (d > 0).sum() / d.size
    return label(vis, f"DEPTH {w}x{h}",
                 f"centre: {centre:.2f} m   valid: {valid:.0f}%   (0-{max_m:g} m, black = no data)")


# --------------------------------------------------------------------------- 4. streaming benchmark

class StreamStats:
    def __init__(self, requested_fps):
        self.requested_fps = requested_fps
        self.count = 0
        self.bytes = 0
        self.first_num = None
        self.last_num = None
        self.drops = 0
        self.first_t = None
        self.last_t = None
        self.frame_bytes = 0

    def add(self, frame, now):
        num = frame.get_frame_number()
        size = frame.get_data_size()
        if self.first_num is None:
            self.first_num, self.first_t = num, now
        elif num > self.last_num + 1:
            self.drops += num - self.last_num - 1
        self.last_num, self.last_t = num, now
        self.count += 1
        self.bytes += size
        self.frame_bytes = size

    def fps(self):
        if self.count < 2 or self.last_t == self.first_t:
            return 0.0
        return (self.count - 1) / (self.last_t - self.first_t)


def build_config(args, serial, with_imu):
    cfg = rs.config()
    cfg.enable_device(serial)
    dw, dh = args.depth
    cw, ch = args.color
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, cw, ch, rs.format.rgb8, args.fps_color or args.fps)
    if not args.no_ir:
        cfg.enable_stream(rs.stream.infrared, 1, dw, dh, rs.format.y8, args.fps)
        cfg.enable_stream(rs.stream.infrared, 2, dw, dh, rs.format.y8, args.fps)
    if with_imu:
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
    return cfg


def start_pipeline(args, serial):
    """Start with IMU; if that fails (e.g. HID not available on macOS), retry without IMU."""
    attempts = [True, False] if not args.no_imu else [False]
    last_err = None
    for with_imu in attempts:
        pipe = rs.pipeline()
        try:
            profile = pipe.start(build_config(args, serial, with_imu))
            if not with_imu and not args.no_imu:
                print("  (IMU could not be started - continuing without accel/gyro)")
            return pipe, profile
        except RuntimeError as e:
            last_err = e
            print(f"  start failed{' with IMU' if with_imu else ''}: {e}")
    sys.exit(f"Could not start the camera: {last_err}\n"
             "Try a lower resolution/fps, a USB 3 port, or run with sudo on macOS.")


def depth_quality(depth_frame, depth_scale):
    d = np.asanyarray(depth_frame.get_data())
    valid = d > 0
    pct = 100.0 * valid.sum() / d.size
    if valid.any():
        m = d[valid] * depth_scale
        return pct, float(m.min()), float(np.median(m)), float(m.max())
    return pct, 0.0, 0.0, 0.0


def save_samples(frames, depth_scale, outdir="camera_samples"):
    os.makedirs(outdir, exist_ok=True)
    for f in frames:
        key = stream_key(f.get_profile())
        if f.is_video_frame():
            img = np.asanyarray(f.get_data())
            np.save(os.path.join(outdir, f"{key}.npy"), img)  # raw data, always
            if cv2 is not None:
                if key == "depth":
                    vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(img, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(outdir, "depth_colormap.png"), vis)
                    cv2.imwrite(os.path.join(outdir, "depth_raw16.png"), img)
                elif key == "color":
                    cv2.imwrite(os.path.join(outdir, "color.png"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(os.path.join(outdir, f"{key}.png"), img)
    print(f"  samples saved to ./{outdir}/ (depth scale = {depth_scale} m/unit)")


def run_benchmark(args, serial):
    print()
    print("=" * 78)
    print(f"STREAMING BENCHMARK  ({args.seconds:.0f} s)")
    print("=" * 78)
    pipe, profile = start_pipeline(args, serial)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print_calibration(profile)

    stats = collections.OrderedDict()
    for s in profile.get_streams():
        stats[stream_key(s)] = StreamStats(s.fps())

    # warm-up: let auto-exposure settle, ignore stats
    for _ in range(int(args.warmup * (args.fps or 30))):
        try:
            pipe.wait_for_frames(2000)
        except RuntimeError:
            break

    quality = []
    saved = False
    framesets = 0
    timeouts = 0
    print()
    print(f"  capturing ... ", end="", flush=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        try:
            fs = pipe.wait_for_frames(2000)
        except RuntimeError:
            timeouts += 1
            continue
        now = time.perf_counter()
        framesets += 1
        for f in fs:
            key = stream_key(f.get_profile())
            if key in stats:
                stats[key].add(f, now)

        depth = fs.get_depth_frame()
        if depth and framesets % 10 == 0:
            quality.append(depth_quality(depth, depth_scale))

        if args.save and not saved and depth and fs.get_color_frame():
            save_samples(list(fs), depth_scale)
            saved = True

        if args.show and cv2 is not None:
            color = fs.get_color_frame()
            if depth and color:
                c = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_RGB2BGR)
                c = label(c, f"COLOUR {c.shape[1]}x{c.shape[0]}")
                d = cv2.resize(depth_view(depth, depth_scale), (c.shape[1], c.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
                if two_view_window("RealSense D455  colour | depth   (q = quit)", c, d) == ord("q"):
                    break
    elapsed = time.perf_counter() - t0
    pipe.stop()
    if args.show and cv2 is not None:
        cv2.destroyAllWindows()
    print("done")

    # ------------------------------------------------------------------ report
    print()
    print("=" * 78)
    print("RESULTS - data from ONE camera")
    print("=" * 78)
    print(f"  {'stream':<12}{'req fps':>8}{'real fps':>10}{'frames':>9}{'dropped':>9}"
          f"{'frame size':>14}{'bandwidth':>16}")
    total_bytes = 0
    for key, s in stats.items():
        bw = s.bytes / elapsed
        total_bytes += s.bytes
        print(f"  {key:<12}{s.requested_fps:>8}{s.fps():>10.1f}{s.count:>9}{s.drops:>9}"
              f"{human_bytes(s.frame_bytes):>14}{human_bytes(bw):>14}/s")
    total_bw = total_bytes / elapsed
    print("  " + "-" * 76)
    print(f"  {'TOTAL':<48}{human_bytes(total_bytes):>14}{human_bytes(total_bw):>14}/s")
    print(f"  = {total_bw * 8 / 1e6:.1f} Mbit/s   |  {framesets} framesets in {elapsed:.1f} s"
          f"  |  timeouts: {timeouts}")
    print(f"  ~ {human_bytes(total_bw * 60).strip()} per minute,"
          f" {human_bytes(total_bw * 3600).strip()} per hour (raw, uncompressed)")

    if "depth" in stats:
        dw, dh = args.depth
        pts_per_s = dw * dh * stats["depth"].fps()
        print(f"\n  Depth points: {dw * dh:,} per frame  ->  ~{pts_per_s / 1e6:.1f} M 3D points/s")

    if quality:
        q = np.array(quality)
        print(f"  Depth valid pixels : {q[:, 0].mean():.1f}% (avg)")
        print(f"  Depth range seen   : min {q[:, 1].min():.2f} m, "
              f"median {np.median(q[:, 2]):.2f} m, max {q[:, 3].max():.2f} m")


# --------------------------------------------------------------------------- 5. OpenCV fallback

def run_opencv(args):
    """Read the D455 RGB sensor as a plain UVC webcam (no librealsense needed)."""
    if cv2 is None:
        sys.exit("--opencv needs opencv: pip install opencv-python")
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.opencv, backend)
    if not cap.isOpened():
        sys.exit(f"Could not open OpenCV camera index {args.opencv}. Try --opencv 1 or 2.")

    req_w, req_h = args.color
    req_fps = args.fps_color or args.fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FPS, req_fps)

    print("=" * 78)
    print(f"OPENCV FALLBACK  (index {args.opencv}, colour only, {args.seconds:.0f} s)")
    print("=" * 78)
    print(f"  requested : {req_w}x{req_h} @ {req_fps} fps")
    print(f"  driver    : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):g} fps")

    # warm-up (auto exposure) and make sure the index actually delivers frames
    got = 0
    t_end = time.perf_counter() + args.warmup
    while time.perf_counter() < t_end:
        ok, _ = cap.read()
        got += ok
    if not got:
        cap.release()
        sys.exit(f"Index {args.opencv} opened but gave no frames "
                 "(probably not the D455 - e.g. an iPhone Continuity Camera). Try another index.")

    print("  capturing ... ", end="", flush=True)
    count, nbytes, failed, shape, saved = 0, 0, 0, None, False
    intervals = []
    t0 = last = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        ok, frame = cap.read()
        now = time.perf_counter()
        if not ok:
            failed += 1
            continue
        if count:
            intervals.append(now - last)
        last = now
        count += 1
        nbytes += frame.nbytes
        shape = frame.shape

        if args.save and not saved:
            os.makedirs("camera_samples", exist_ok=True)
            cv2.imwrite(os.path.join("camera_samples", "color_opencv.png"), frame)
            saved = True
        if args.show:
            left = label(frame.copy(), f"COLOUR {frame.shape[1]}x{frame.shape[0]}  (OpenCV)")
            right = np.zeros_like(frame)
            label(right, "DEPTH: not available in OpenCV mode",
                  "needs librealsense:  ./run_camera.sh  (with admin password)")
            if two_view_window("D455  colour | depth   (q = quit)", left, right) == ord("q"):
                break
    elapsed = time.perf_counter() - t0
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    print("done")

    fps = count / elapsed if elapsed else 0
    bw = nbytes / elapsed if elapsed else 0
    h, w = shape[:2] if shape is not None else (0, 0)
    print()
    print("=" * 78)
    print("RESULTS - colour stream via OpenCV")
    print("=" * 78)
    print(f"  resolution     : {w}x{h} (BGR, 3 bytes/pixel)")
    print(f"  frames         : {count} in {elapsed:.1f} s  ->  {fps:.1f} fps   (read failures: {failed})")
    if intervals:
        iv = np.array(intervals) * 1000
        print(f"  frame interval : mean {iv.mean():.1f} ms, max {iv.max():.1f} ms "
              f"(gaps > 1.5x mean: {(iv > 1.5 * iv.mean()).sum()})")
    print(f"  frame size     : {human_bytes(shape[0] * shape[1] * shape[2] if shape else 0).strip()}")
    print(f"  bandwidth      : {human_bytes(bw).strip()}/s  = {bw * 8 / 1e6:.1f} Mbit/s (decoded)")
    print(f"  ~ {human_bytes(bw * 60).strip()} per minute, {human_bytes(bw * 3600).strip()} per hour")
    if saved:
        print("  sample saved to ./camera_samples/color_opencv.png")
    print("\n  Note: depth, infrared and IMU need librealsense (run without --opencv).")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Intel RealSense D455 data-throughput test")
    ap.add_argument("--seconds", type=float, default=10, help="capture duration")
    ap.add_argument("--warmup", type=float, default=2, help="warm-up seconds (not measured)")
    ap.add_argument("--depth", type=parse_res, default=(848, 480), help="depth/IR resolution WxH")
    ap.add_argument("--color", type=parse_res, default=(1280, 720), help="color resolution WxH")
    ap.add_argument("--fps", type=int, default=30, help="depth/IR fps")
    ap.add_argument("--fps-color", type=int, default=0, help="color fps (default = --fps)")
    ap.add_argument("--no-ir", action="store_true", help="disable left/right infrared streams")
    ap.add_argument("--no-imu", action="store_true", help="disable accel/gyro")
    ap.add_argument("--save", action="store_true", help="save one sample of each stream")
    ap.add_argument("--show", action="store_true", help="live preview (needs opencv)")
    ap.add_argument("--list-only", action="store_true", help="only list capabilities")
    ap.add_argument("--serial", default=None, help="serial number if several cameras")
    ap.add_argument("--opencv", type=int, nargs="?", const=0, default=None, metavar="INDEX",
                    help="fallback: read only the RGB stream via OpenCV webcam INDEX (default 0)")
    args = ap.parse_args()

    if args.opencv is not None:
        run_opencv(args)
        return

    if rs is None:
        sys.exit("pyrealsense2 is not installed.\n"
                 "  macOS:         pip install pyrealsense2-macosx numpy\n"
                 "  Linux/Windows: pip install pyrealsense2 numpy\n"
                 "  (or use --opencv for the colour stream only)")

    if (args.save or args.show) and cv2 is None:
        print("note: opencv not installed - --show disabled, --save writes .npy only")

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        sys.exit("No RealSense device found. Check the USB 3 cable "
                 "(on macOS, try: sudo python3 camera.py).")
    print(f"Found {len(devices)} RealSense device(s)")

    dev = devices[0]
    if args.serial:
        match = [d for d in devices if safe_info(d, rs.camera_info.serial_number) == args.serial]
        if not match:
            sys.exit(f"No device with serial {args.serial}")
        dev = match[0]
    serial = safe_info(dev, rs.camera_info.serial_number)

    print_device_info(dev)
    print_sensors_and_profiles(dev)
    if not args.list_only:
        run_benchmark(args, serial)


if __name__ == "__main__":
    main()
