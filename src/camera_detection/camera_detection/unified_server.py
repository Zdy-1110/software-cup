"""
统一相机视频与目标检测服务。

单进程只打开一次摄像头，一条 GStreamer 管线分成两支：
- :8765 发送 VJPG 帧头 + 相机原始 JPEG（1080p30）
- :8766 发送与视频共享 frame_id/PTS 的检测 JSON
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import struct
import threading
import time
from collections import OrderedDict
from typing import Dict

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import rclpy
import websockets
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu

from camera_detection.detection_server import (
    CLASS_NAMES,
    CONF_THRESH,
    INFER_HEIGHT,
    INFER_WIDTH,
    NMS_THRESH,
    RKNNRuntime,
    decode_ppyoloe,
)
from camera_detection.intelligence import (
    CloudGenerator,
    ConfidencePolicy,
    HudCardGenerator,
    IoUTracker,
    LANDMARK_CARDS,
    SceneEngine,
    TemplateGenerator,
    VisionUnderstandingClient,
)

logging.basicConfig(level=logging.INFO, format='[unified] %(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('unified_server')

VIDEO_DEVICE = os.environ.get('VIDEO_DEVICE', '/dev/video20')
WIDTH = int(os.environ.get('WIDTH', '1920'))
HEIGHT = int(os.environ.get('HEIGHT', '1080'))
FRAMERATE = int(os.environ.get('FRAMERATE', '30'))
WS_PORT = int(os.environ.get('WS_PORT', '8765'))
DETECTION_PORT = int(os.environ.get('DETECTION_PORT', '8766'))
DETECTION_FPS = float(os.environ.get('DETECTION_FPS', '30'))
TELEMETRY_FPS = float(os.environ.get('TELEMETRY_FPS', '20'))
IMU_TOPIC = os.environ.get('IMU_TOPIC', '/imu')
ODOM_TOPIC = os.environ.get('ODOM_TOPIC', '/odom_raw')
IMU_ACCEL_UNIT = os.environ.get('IMU_ACCEL_UNIT', 'auto').lower()
RKNN_MODEL = os.environ.get(
    'RKNN_MODEL',
    '/home/teamhd/Downloads/ppyoloe_carrace_rk3588_official_split_int8_416.rknn',
)
CLOUD_API_URL = os.environ.get('CLOUD_API_URL', '')
CLOUD_API_KEY = os.environ.get('CLOUD_API_KEY', '')
CLOUD_API_MODEL = os.environ.get('CLOUD_API_MODEL', '')
CLOUD_API_TIMEOUT = float(os.environ.get('CLOUD_API_TIMEOUT', '2.5'))
UNDERSTANDING_API_URL = os.environ.get(
    'UNDERSTANDING_API_URL', 'https://qianfan.baidubce.com/v2')
UNDERSTANDING_API_KEY = os.environ.get('UNDERSTANDING_API_KEY', '')
UNDERSTANDING_MODEL = os.environ.get(
    'UNDERSTANDING_MODEL', 'ernie-4.5-turbo-vl')
UNDERSTANDING_TIMEOUT = float(os.environ.get('UNDERSTANDING_TIMEOUT', '3.0'))
UNDERSTANDING_CONF_MIN = float(os.environ.get('UNDERSTANDING_CONF_MIN', '0.25'))
UNDERSTANDING_CONF_MAX = float(os.environ.get('UNDERSTANDING_CONF_MAX', '0.60'))
UNDERSTANDING_COOLDOWN = float(os.environ.get('UNDERSTANDING_COOLDOWN', '30'))
HUD_API_URL = os.environ.get(
    'HUD_API_URL', 'https://qianfan.baidubce.com/v2')
HUD_API_KEY = os.environ.get('HUD_API_KEY', '')
HUD_API_MODEL = os.environ.get('HUD_API_MODEL', 'ernie-5.1')
HUD_API_TIMEOUT = float(os.environ.get('HUD_API_TIMEOUT', '0.8'))
HUD_REPEAT_INTERVAL = float(os.environ.get('HUD_REPEAT_INTERVAL', '3.0'))
try:
    CLASS_DISPLAY_NAMES = json.loads(os.environ.get('CLASS_DISPLAY_NAMES', '{}'))
except json.JSONDecodeError:
    CLASS_DISPLAY_NAMES = {}


class TelemetryBridge(Node):
    """订阅底盘反馈，并向WebSocket侧提供线程安全的最新快照。"""

    GRAVITY = 9.80665

    def __init__(self):
        super().__init__('camera_telemetry_bridge')
        self._lock = threading.Lock()
        self._speed_mps = 0.0
        self._accel_mps2 = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self._imu_updated_at = 0.0
        self._odom_updated_at = 0.0
        self.create_subscription(Imu, IMU_TOPIC, self._on_imu, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, 10)
        log.info('ROS telemetry subscriptions imu=%s odom=%s accel_unit=%s',
                 IMU_TOPIC, ODOM_TOPIC, IMU_ACCEL_UNIT)

    @staticmethod
    def _accel_scale(x: float, y: float, z: float) -> float:
        if IMU_ACCEL_UNIT == 'g':
            return TelemetryBridge.GRAVITY
        if IMU_ACCEL_UNIT == 'mps2':
            return 1.0
        magnitude = (x * x + y * y + z * z) ** 0.5
        return TelemetryBridge.GRAVITY if 0.5 <= magnitude <= 2.0 else 1.0

    def _on_imu(self, msg: Imu):
        x = float(msg.linear_acceleration.x)
        y = float(msg.linear_acceleration.y)
        z = float(msg.linear_acceleration.z)
        scale = self._accel_scale(x, y, z)
        with self._lock:
            self._accel_mps2 = {'x': x * scale, 'y': y * scale, 'z': z * scale}
            self._imu_updated_at = time.time()

    def _on_odom(self, msg: Odometry):
        with self._lock:
            self._speed_mps = float(msg.twist.twist.linear.x)
            self._odom_updated_at = time.time()

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            accel = dict(self._accel_mps2)
            speed = self._speed_mps
            imu_age = now - self._imu_updated_at if self._imu_updated_at else None
            odom_age = now - self._odom_updated_at if self._odom_updated_at else None
        return {
            'type': 'telemetry',
            'stamp_ms': round(now * 1000),
            'motor_speed_mps': round(speed, 4),
            'imu_accel_mps2': {key: round(value, 4) for key, value in accel.items()},
            'imu_stale': imu_age is None or imu_age > 0.5,
            'motor_speed_stale': odom_age is None or odom_age > 0.5,
        }


class FrameRegistry:
    """将源帧 PTS 稳定映射为视频与检测共用的 frame_id。"""

    def __init__(self, max_entries: int = 256):
        self._lock = threading.Lock()
        self._next_id = 1
        self._by_pts: OrderedDict[int, int] = OrderedDict()
        self._max_entries = max_entries

    def resolve(self, pts_ns: int) -> int:
        with self._lock:
            frame_id = self._by_pts.get(pts_ns)
            if frame_id is None:
                frame_id = self._next_id
                self._next_id += 1
                self._by_pts[pts_ns] = frame_id
                while len(self._by_pts) > self._max_entries:
                    self._by_pts.popitem(last=False)
            return frame_id


class UnifiedServer:
    MAGIC = b'VJPG'
    HEADER_STRUCT = struct.Struct('!4sQQI')

    def __init__(self):
        Gst.init(None)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pipeline: Gst.Pipeline | None = None
        self._video_clients: set = set()
        self._detection_clients: set = set()
        self._clients_lock = threading.Lock()
        self._registry = FrameRegistry()
        self._rknn: RKNNRuntime | None = None
        self._tracker = IoUTracker()
        self._scene_engine = SceneEngine()
        self._template_generator = TemplateGenerator()
        self._cloud_generator = CloudGenerator(
            CLOUD_API_URL, CLOUD_API_KEY, CLOUD_API_MODEL, CLOUD_API_TIMEOUT)
        self._cloud_executor = ThreadPoolExecutor(max_workers=1)
        self._cloud_slots = threading.BoundedSemaphore(2)
        self._understanding = VisionUnderstandingClient(
            UNDERSTANDING_API_URL, UNDERSTANDING_API_KEY,
            UNDERSTANDING_MODEL, UNDERSTANDING_TIMEOUT)
        self._confidence_policy = ConfidencePolicy(
            UNDERSTANDING_CONF_MIN, UNDERSTANDING_CONF_MAX)
        self._understanding_executor = ThreadPoolExecutor(max_workers=1)
        self._understanding_slot = threading.BoundedSemaphore(1)
        self._understanding_last: dict[int, float] = {}
        self._hud_generator = HudCardGenerator(
            HUD_API_URL, HUD_API_KEY, HUD_API_MODEL, HUD_API_TIMEOUT)
        self._hud_executor = ThreadPoolExecutor(max_workers=1)
        self._hud_slots = threading.BoundedSemaphore(2)
        self._hud_last_by_class: dict[str, float] = {}
        self._jpeg_lock = threading.Lock()
        self._jpeg_by_frame: OrderedDict[int, bytes] = OrderedDict()
        self._scene_revision = 0
        self._active_track_ids: set[int] = set()
        self._latest_telemetry: dict = {}
        self._running = False

        self._frame_lock = threading.Condition()
        self._latest_detection_frame: tuple[int, int, float, np.ndarray] | None = None
        self._detection_sequence = 0
        self._infer_thread: threading.Thread | None = None
        self._telemetry_node: TelemetryBridge | None = None
        self._ros_thread: threading.Thread | None = None

        self._video_send_pending = False
        self._detection_send_pending = False
        self._video_count = 0
        self._video_sent = 0
        self._video_dropped = 0
        self._infer_count = 0
        self._timing_sums: Dict[str, float] = {}
        self._last_stats_ts = time.time()
        self._probe_logged = False

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def _build_pipeline(self) -> str:
        return (
            f'v4l2src device={VIDEO_DEVICE} do-timestamp=true '
            f'! image/jpeg,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/1 '
            '! jpegparse ! tee name=t '
            't. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream '
            '! appsink name=video_sink emit-signals=true max-buffers=1 drop=true sync=false '
            't. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream '
            '! mppjpegdec format=NV12 fast-mode=true ignore-error=true '
            f'! videoscale ! video/x-raw,format=NV12,width={INFER_WIDTH},height={INFER_HEIGHT} '
            '! videoconvert '
            f'! video/x-raw,format=RGB,width={INFER_WIDTH},height={INFER_HEIGHT} '
            '! appsink name=detect_sink emit-signals=true max-buffers=1 drop=true sync=false'
        )

    def start(self):
        if not rclpy.ok():
            rclpy.init(args=None)
        self._telemetry_node = TelemetryBridge()
        self._ros_thread = threading.Thread(
            target=rclpy.spin, args=(self._telemetry_node,), daemon=True)
        self._ros_thread.start()
        log.info('Loading RKNN model: %s', RKNN_MODEL)
        self._rknn = RKNNRuntime(RKNN_MODEL)
        pipeline_str = self._build_pipeline()
        log.info('Pipeline: %s', pipeline_str)
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._pipeline.get_by_name('video_sink').connect('new-sample', self._on_video_sample)
        self._pipeline.get_by_name('detect_sink').connect('new-sample', self._on_detection_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::error', self._on_error)
        bus.connect('message::eos', self._on_eos)
        glib_loop = GLib.MainLoop()
        threading.Thread(target=glib_loop.run, daemon=True).start()

        self._running = True
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Failed to start unified GStreamer pipeline')
        log.info('Unified pipeline PLAYING %s %dx%d@%dfps',
                 VIDEO_DEVICE, WIDTH, HEIGHT, FRAMERATE)
        log.info('Cloud generation enabled=%s model=%s',
             self._cloud_generator.enabled, CLOUD_API_MODEL or '-')
        log.info('Visual understanding enabled=%s model=%s gray_zone=%.2f..%.2f',
             self._understanding.enabled, UNDERSTANDING_MODEL,
             UNDERSTANDING_CONF_MIN, UNDERSTANDING_CONF_MAX)

    @staticmethod
    def _buffer_pts(buf: Gst.Buffer) -> int:
        return int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else 0

    def _on_video_sample(self, appsink) -> Gst.FlowReturn:
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        jpeg = bytes(mapinfo.data)
        pts_ns = self._buffer_pts(buf)
        buf.unmap(mapinfo)

        frame_id = self._registry.resolve(pts_ns)
        with self._jpeg_lock:
            self._jpeg_by_frame[frame_id] = jpeg
            while len(self._jpeg_by_frame) > 12:
                self._jpeg_by_frame.popitem(last=False)
        packet = self.HEADER_STRUCT.pack(self.MAGIC, frame_id, pts_ns, len(jpeg)) + jpeg
        self._video_count += 1

        if not self._probe_logged:
            log.info('probe frame_id=%d pts_ns=%d jpeg=%d soi=%s eoi=%s',
                     frame_id, pts_ns, len(jpeg), jpeg[:2].hex(), jpeg[-2:].hex())
            self._probe_logged = True

        if self._loop and self._video_clients:
            if self._video_send_pending:
                self._video_dropped += 1
            else:
                with self._clients_lock:
                    clients = list(self._video_clients)
                self._video_send_pending = True
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_video(packet, clients), self._loop)
                self._video_sent += 1
        self._maybe_log_stats()
        return Gst.FlowReturn.OK

    def _on_detection_sample(self, appsink) -> Gst.FlowReturn:
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
            INFER_HEIGHT, INFER_WIDTH, 3).copy()
        pts_ns = self._buffer_pts(buf)
        buf.unmap(mapinfo)
        frame_id = self._registry.resolve(pts_ns)

        with self._frame_lock:
            self._detection_sequence += 1
            self._latest_detection_frame = (frame_id, pts_ns, time.time(), frame)
            self._frame_lock.notify_all()
        return Gst.FlowReturn.OK

    def _infer_loop(self):
        target_interval = 1.0 / max(DETECTION_FPS, 1.0)
        consumed_sequence = 0
        while self._running:
            wait_start = time.perf_counter()
            with self._frame_lock:
                while self._running and self._detection_sequence == consumed_sequence:
                    self._frame_lock.wait(0.1)
                if not self._running:
                    return
                consumed_sequence = self._detection_sequence
                item = self._latest_detection_frame
            wait_ms = (time.perf_counter() - wait_start) * 1000
            if item is None:
                continue
            frame_id, pts_ns, capture_ts, frame = item

            infer_start = time.perf_counter()
            try:
                outputs, timings = self._rknn.infer(frame)
                post_start = time.perf_counter()
                detections = decode_ppyoloe(
                    outputs, CONF_THRESH, NMS_THRESH, WIDTH, HEIGHT)
                timings['post_ms'] = (time.perf_counter() - post_start) * 1000
            except Exception as exc:
                log.warning('Inference error: %s', exc)
                detections = []
                timings = {}
            latency_ms = (time.perf_counter() - infer_start) * 1000
            timings['wait_ms'] = wait_ms
            timings['infer_total_ms'] = latency_ms
            self._infer_count += 1
            for key, value in timings.items():
                self._timing_sums[key] = self._timing_sums.get(key, 0.0) + value

            tracks = self._tracker.update(detections)
            scene, events = self._scene_engine.update(
                self._tracker.active(), frame_id, self._latest_telemetry)
            self._scene_revision = scene['revision']
            self._active_track_ids = {
                track['track_id'] for track in scene['tracks']}
            self._maybe_submit_understanding(tracks, frame_id, scene['revision'])
            messages = [json.dumps(self._build_detection_payload(
                tracks, latency_ms, frame_id, pts_ns, capture_ts))]
            messages.append(json.dumps(scene))
            for event in events:
                messages.append(json.dumps(event))
                messages.append(json.dumps(
                    self._template_generator.generate(event, scene['revision'])))
                self._submit_cloud_generation(event, scene)
                if event['event_type'] == 'object_entered':
                    track = next((item for item in scene['tracks']
                                  if item['track_id'] == event['track_id']), None)
                    if (track and self._confidence_policy.state(
                            track['confidence']) == 'accepted'):
                        self._submit_hud_card({
                            'event_id': event['event_id'],
                            'track_id': track['track_id'],
                            'scene_revision': scene['revision'],
                            'class_name': track['class_name'],
                            'display_name': self._display_name(track['class_name']),
                            'confidence': track['confidence'],
                            'telemetry': scene.get('telemetry', {}),
                        })
            if self._loop and self._detection_clients and not self._detection_send_pending:
                with self._clients_lock:
                    clients = list(self._detection_clients)
                self._detection_send_pending = True
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_detection(messages, clients), self._loop)

            elapsed = time.perf_counter() - infer_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            self._maybe_log_stats()

    def _build_detection_payload(self, detections: list, latency_ms: float,
                                 frame_id: int, pts_ns: int, capture_ts: float) -> dict:
        now = time.time()
        sec = int(now)
        cap_sec = int(capture_ts)
        return {
            'type': 'detection',
            'version': 2,
            'stamp': {'sec': sec, 'nanosec': int((now - sec) * 1e9)},
            'frame_id': frame_id,
            'frame_pts_ns': pts_ns,
            'capture_stamp': {
                'sec': cap_sec,
                'nanosec': int((capture_ts - cap_sec) * 1e9),
            },
            'latency_ms': round(latency_ms, 2),
            'input_size': [WIDTH, HEIGHT],
            'detections': [
                {
                    'id': d.track_id,
                    'track_id': d.track_id,
                    'class_name': d.class_name,
                    'bbox': d.bbox,
                    'confidence': d.confidence,
                    'raw_confidence': d.raw_confidence,
                    'track_confidence': d.confidence,
                    'confidence_state': self._confidence_policy.state(
                        d.confidence),
                }
                for d in detections
            ],
        }

    async def _broadcast_video(self, packet: bytes, clients: list):
        try:
            await asyncio.gather(
                *[self._safe_send(ws, packet) for ws in clients],
                return_exceptions=True)
        finally:
            self._video_send_pending = False

    async def _broadcast_detection(self, messages: list[str], clients: list):
        try:
            for message in messages:
                await asyncio.gather(
                    *[self._safe_send(ws, message) for ws in clients],
                    return_exceptions=True)
        finally:
            self._detection_send_pending = False

    @staticmethod
    async def _safe_send(ws, data):
        await ws.send(data)

    def _submit_cloud_generation(self, event: dict, scene: dict):
        if not self._cloud_generator.enabled or not self._cloud_slots.acquire(False):
            return

        def task():
            try:
                result = self._cloud_generator.generate(event, scene)
                if not result or not self._loop:
                    return
                track_id = result.get('track_id')
                if (track_id is not None and track_id not in self._active_track_ids
                        or self._scene_revision - result['scene_revision'] > 60):
                    return
                with self._clients_lock:
                    clients = list(self._detection_clients)
                if clients:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_auxiliary(json.dumps(result), clients), self._loop)
            except Exception as exc:
                log.warning('Cloud generation error: %s', exc)
            finally:
                self._cloud_slots.release()

        self._cloud_executor.submit(task)

    async def _broadcast_auxiliary(self, message: str, clients: list):
        await asyncio.gather(
            *[self._safe_send(ws, message) for ws in clients],
            return_exceptions=True)

    @staticmethod
    def _display_name(class_name: str) -> str:
        value = CLASS_DISPLAY_NAMES.get(
            class_name, LANDMARK_CARDS.get(class_name, {}).get(
                'display_name', class_name))
        return str(value).strip()[:24] or class_name

    def _submit_hud_card(self, context: dict):
        now = time.monotonic()
        class_name = context['class_name']
        if (now - self._hud_last_by_class.get(class_name, 0.0)
                < HUD_REPEAT_INTERVAL or not self._hud_slots.acquire(False)):
            return
        self._hud_last_by_class[class_name] = now
        context['reference'] = dict(LANDMARK_CARDS.get(class_name, {}))

        def task():
            try:
                try:
                    card = self._hud_generator.generate(context)
                except Exception as exc:
                    log.warning('HUD generation error, using template: %s', exc)
                    card = self._hud_generator.template(context)
                if not self._loop or context['track_id'] not in self._active_track_ids:
                    return
                with self._clients_lock:
                    clients = list(self._detection_clients)
                if clients:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_auxiliary(json.dumps(card), clients),
                        self._loop)
            finally:
                self._hud_slots.release()

        self._hud_executor.submit(task)

    def _maybe_submit_understanding(self, tracks, frame_id: int, revision: int):
        if not self._understanding.enabled:
            return
        now = time.monotonic()
        candidates = [
            track for track in tracks
            if track.confirmed
            and self._confidence_policy.state(track.confidence) == 'review'
            and now - self._understanding_last.get(
                track.track_id, 0) >= UNDERSTANDING_COOLDOWN
        ]
        candidate = min(
            candidates,
            key=lambda track: abs(track.confidence - UNDERSTANDING_CONF_MAX),
            default=None)
        if candidate is None or not self._understanding_slot.acquire(False):
            return
        with self._jpeg_lock:
            jpeg = self._jpeg_by_frame.get(frame_id)
        if jpeg is None:
            self._understanding_slot.release()
            return
        self._understanding_last[candidate.track_id] = now
        payload = {
            'track_id': candidate.track_id, 'class_name': candidate.class_name,
            'confidence': candidate.confidence,
            'raw_confidence': candidate.raw_confidence,
            'track_confidence': candidate.confidence,
            'bbox': candidate.bbox,
        }

        def task():
            try:
                result = self._understanding.confirm(jpeg, payload, CLASS_NAMES)
                if not result or not self._loop:
                    return
                fusion = self._confidence_policy.fuse(payload, result)
                message = {
                    'type': 'understanding', 'version': 2,
                    'frame_id': frame_id, 'track_id': candidate.track_id,
                    'scene_revision': revision, 'source': UNDERSTANDING_MODEL,
                    'candidate': payload, 'result': result,
                    **fusion,
                    'stamp_ms': round(time.time() * 1000),
                }
                if candidate.track_id not in self._active_track_ids:
                    return
                if fusion['decision'] in ('accepted', 'reclassified'):
                    self._submit_hud_card({
                        'event_id': None,
                        'track_id': candidate.track_id,
                        'scene_revision': revision,
                        'class_name': fusion['effective_class_name'],
                        'display_name': self._display_name(
                            fusion['effective_class_name']),
                        'confidence': fusion['effective_confidence'],
                        'telemetry': dict(self._latest_telemetry),
                    })
                with self._clients_lock:
                    clients = list(self._detection_clients)
                if clients:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_auxiliary(json.dumps(message), clients),
                        self._loop)
            except Exception as exc:
                log.warning('Visual understanding error: %s', exc)
            finally:
                self._understanding_slot.release()

        self._understanding_executor.submit(task)

    def add_video_client(self, ws):
        with self._clients_lock:
            self._video_clients.add(ws)
        log.info('video client connected total=%d', len(self._video_clients))

    def remove_video_client(self, ws):
        with self._clients_lock:
            self._video_clients.discard(ws)

    def add_detection_client(self, ws):
        with self._clients_lock:
            self._detection_clients.add(ws)
        log.info('detection client connected total=%d', len(self._detection_clients))

    def remove_detection_client(self, ws):
        with self._clients_lock:
            self._detection_clients.discard(ws)

    def _maybe_log_stats(self):
        now = time.time()
        elapsed = now - self._last_stats_ts
        if elapsed < 5.0:
            return
        infer_count = max(self._infer_count, 1)
        avg = lambda key: self._timing_sums.get(key, 0.0) / infer_count
        log.info(
            'stats video_in=%.1ffps video_sent=%.1ffps video_drop=%d '
            'infer=%.1ffps wait=%.1fms npu=%.1fms total=%.1fms clients=%d/%d',
            self._video_count / elapsed,
            self._video_sent / elapsed,
            self._video_dropped,
            self._infer_count / elapsed,
            avg('wait_ms'), avg('npu_ms'), avg('infer_total_ms'),
            len(self._video_clients), len(self._detection_clients))
        self._video_count = 0
        self._video_sent = 0
        self._video_dropped = 0
        self._infer_count = 0
        self._timing_sums.clear()
        self._last_stats_ts = now

    def _on_error(self, bus, message):
        err, debug = message.parse_error()
        log.error('GStreamer error: %s debug=%s', err, debug)

    def _on_eos(self, bus, message):
        log.warning('GStreamer EOS received')

    def stop(self):
        self._running = False
        with self._frame_lock:
            self._frame_lock.notify_all()
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._rknn:
            self._rknn.close()
        self._cloud_executor.shutdown(wait=False, cancel_futures=True)
        self._understanding_executor.shutdown(wait=False, cancel_futures=True)
        if rclpy.ok():
            rclpy.shutdown()
        if self._telemetry_node:
            self._telemetry_node.destroy_node()
            self._telemetry_node = None

    async def telemetry_loop(self):
        interval = 1.0 / max(TELEMETRY_FPS, 1.0)
        while self._running:
            if self._telemetry_node and self._detection_clients:
                self._latest_telemetry = self._telemetry_node.snapshot()
                message = json.dumps(self._latest_telemetry)
                with self._clients_lock:
                    clients = list(self._detection_clients)
                await asyncio.gather(
                    *[self._safe_send(ws, message) for ws in clients],
                    return_exceptions=True)
            await asyncio.sleep(interval)


server = UnifiedServer()


async def video_handler(websocket):
    server.add_video_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        server.remove_video_client(websocket)


async def detection_handler(websocket):
    server.add_detection_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        server.remove_detection_client(websocket)


async def serve():
    server.set_event_loop(asyncio.get_running_loop())
    server.start()
    async with websockets.serve(
        video_handler, '0.0.0.0', WS_PORT, compression=None, max_size=None
    ), websockets.serve(
        detection_handler, '0.0.0.0', DETECTION_PORT, compression=None
    ):
        log.info('Video WebSocket ws://0.0.0.0:%d', WS_PORT)
        log.info('Detection WebSocket ws://0.0.0.0:%d', DETECTION_PORT)
        telemetry_task = asyncio.create_task(server.telemetry_loop())
        try:
            await asyncio.Future()
        finally:
            telemetry_task.cancel()


def main():
    try:
        asyncio.run(serve())
    finally:
        server.stop()


if __name__ == '__main__':
    main()
