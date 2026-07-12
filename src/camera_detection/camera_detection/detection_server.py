"""
检测数据 WebSocket 服务
- 从摄像头取帧 → 缩放到 416×416 → RKNN ppyoloe 推理
- 结果以 JSON 广播到 ws://0.0.0.0:8766
- JSON 结构与前端协议完全匹配
"""

import asyncio
import ctypes
import json
import logging
import os
import threading
import time
from collections import namedtuple
from typing import Dict, List

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
import websockets

logging.basicConfig(level=logging.INFO, format='[detect] %(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('detection_server')

# ── 配置 ──────────────────────────────────────────────────────────────────
VIDEO_DEVICE    = os.environ.get('VIDEO_DEVICE',    '/dev/video20')
VIDEO_SOURCE    = os.environ.get('VIDEO_SOURCE',    'shm')
DETECT_SHM_SOCKET = os.environ.get('DETECT_SHM_SOCKET', '/tmp/camera_detect.shm')
RKNN_MODEL      = os.environ.get('RKNN_MODEL',
                  '/home/teamhd/Downloads/ppyoloe_carrace_rk3588_official_split_int8_416.rknn')
DETECTION_PORT  = int(os.environ.get('DETECTION_PORT', '8766'))
DETECTION_FPS   = float(os.environ.get('DETECTION_FPS', '30'))
RKNN_INPUT_MODE = os.environ.get('RKNN_INPUT_MODE', 'uint8').lower()
SOURCE_WIDTH    = int(os.environ.get('WIDTH',  '1920'))
SOURCE_HEIGHT   = int(os.environ.get('HEIGHT', '1080'))
INFER_WIDTH     = 416
INFER_HEIGHT    = 416
_SCALE_X        = SOURCE_WIDTH / INFER_WIDTH
_SCALE_Y        = SOURCE_HEIGHT / INFER_HEIGHT
CONF_THRESH     = float(os.environ.get('CONF_THRESH', '0.3'))
NMS_THRESH      = float(os.environ.get('NMS_THRESH',  '0.45'))
# ppyoloe 输出类别（根据实际模型调整）
CLASS_NAMES     = os.environ.get('CLASS_NAMES',
                  'bm,cjl,jsjd,jzt,lu,mtl,nc,tt,ydm,zynsx').split(',')

Detection = namedtuple('Detection', ['id', 'class_name', 'bbox', 'confidence'])


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [('n_input', ctypes.c_uint32), ('n_output', ctypes.c_uint32)]


class RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ('index', ctypes.c_uint32), ('n_dims', ctypes.c_uint32),
        ('dims', ctypes.c_uint32 * 16), ('name', ctypes.c_char * 256),
        ('n_elems', ctypes.c_uint32), ('size', ctypes.c_uint32),
        ('fmt', ctypes.c_int32), ('type', ctypes.c_int32),
        ('qnt_type', ctypes.c_int32), ('fl', ctypes.c_int8),
        ('zp', ctypes.c_int32), ('scale', ctypes.c_float),
        ('w_stride', ctypes.c_uint32), ('size_with_stride', ctypes.c_uint32),
        ('pass_through', ctypes.c_uint8), ('h_stride', ctypes.c_uint32),
    ]


class RknnInput(ctypes.Structure):
    _fields_ = [
        ('index', ctypes.c_uint32), ('buf', ctypes.c_void_p),
        ('size', ctypes.c_uint32), ('pass_through', ctypes.c_uint8),
        ('type', ctypes.c_int32), ('fmt', ctypes.c_int32),
    ]


class RknnOutput(ctypes.Structure):
    _fields_ = [
        ('want_float', ctypes.c_uint8), ('is_prealloc', ctypes.c_uint8),
        ('index', ctypes.c_uint32), ('buf', ctypes.c_void_p),
        ('size', ctypes.c_uint32),
    ]


class RknnPerfRun(ctypes.Structure):
    _fields_ = [('run_duration', ctypes.c_int64)]


class RknnSdkVersion(ctypes.Structure):
    _fields_ = [('api_version', ctypes.c_char * 256), ('drv_version', ctypes.c_char * 256)]


# ── RKNN Lite 封装（通过 ctypes 调用 librknnrt.so）────────────────────────

class RKNNRuntime:
    """轻量封装 librknnrt C API，无需 rknn_toolkit_lite Python 包"""

    RKNN_SUCC = 0
    RKNN_QUERY_IN_OUT_NUM = 0
    RKNN_QUERY_INPUT_ATTR = 1
    RKNN_QUERY_OUTPUT_ATTR = 2
    RKNN_QUERY_PERF_RUN = 4
    RKNN_QUERY_SDK_VERSION = 5
    RKNN_TENSOR_FLOAT32 = 0
    RKNN_TENSOR_UINT8 = 3
    RKNN_TENSOR_NHWC = 1
    RKNN_NPU_CORE_0_1_2 = 7

    def __init__(self, model_path: str):
        self._lib = ctypes.CDLL('/usr/lib/librknnrt.so')
        self._ctx = None
        self._num_inputs = 0
        self._num_outputs = 0
        self._input_type = self.RKNN_TENSOR_FLOAT32
        self._setup_prototypes()
        self._load_model(model_path)

    def _setup_prototypes(self):
        lib = self._lib
        # rknn_init
        lib.rknn_init.restype  = ctypes.c_int
        lib.rknn_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # context
            ctypes.c_void_p,                   # model data
            ctypes.c_uint32,                   # model size
            ctypes.c_uint32,                   # flags
            ctypes.c_void_p,                   # extend
        ]
        # rknn_inputs_set
        lib.rknn_inputs_set.restype  = ctypes.c_int
        lib.rknn_inputs_set.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p
        ]
        # rknn_run
        lib.rknn_run.restype  = ctypes.c_int
        lib.rknn_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        # rknn_outputs_get
        lib.rknn_outputs_get.restype  = ctypes.c_int
        lib.rknn_outputs_get.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p
        ]
        # rknn_outputs_release
        lib.rknn_outputs_release.restype  = ctypes.c_int
        lib.rknn_outputs_release.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p
        ]
        # rknn_destroy
        lib.rknn_destroy.restype  = ctypes.c_int
        lib.rknn_destroy.argtypes = [ctypes.c_void_p]
        # rknn_set_core_mask
        lib.rknn_set_core_mask.restype  = ctypes.c_int
        lib.rknn_set_core_mask.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        # rknn_query (用于获取输出 tensor 信息)
        lib.rknn_query.restype  = ctypes.c_int
        lib.rknn_query.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32
        ]

    def _load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            model_data = f.read()
        buf = ctypes.create_string_buffer(model_data)
        ctx = ctypes.c_void_p()
        ret = self._lib.rknn_init(
            ctypes.byref(ctx),
            buf, len(model_data),
            0, None
        )
        if ret != self.RKNN_SUCC:
            raise RuntimeError(f'rknn_init failed: {ret}')
        self._ctx = ctx
        self._log_model_info()
        ret = self._lib.rknn_set_core_mask(self._ctx, self.RKNN_NPU_CORE_0_1_2)
        if ret != self.RKNN_SUCC:
            log.warning('rknn_set_core_mask(0_1_2) failed: %s', ret)
        else:
            log.info('RKNN NPU core mask: 0_1_2')
        log.info('RKNN model loaded: %s', model_path)

    def _query(self, cmd: int, info) -> None:
        ret = self._lib.rknn_query(self._ctx, cmd, ctypes.byref(info), ctypes.sizeof(info))
        if ret != self.RKNN_SUCC:
            raise RuntimeError(f'rknn_query({cmd}) failed: {ret}')

    @staticmethod
    def _tensor_desc(attr: RknnTensorAttr) -> str:
        dims = list(attr.dims[:attr.n_dims])
        name = attr.name.split(b'\0', 1)[0].decode(errors='replace')
        return (f'index={attr.index} name={name} dims={dims} fmt={attr.fmt} '
                f'type={attr.type} qnt={attr.qnt_type} zp={attr.zp} scale={attr.scale:.6g} '
                f'size={attr.size} stride_size={attr.size_with_stride}')

    def _log_model_info(self):
        try:
            version = RknnSdkVersion()
            self._query(self.RKNN_QUERY_SDK_VERSION, version)
            log.info('RKNN SDK api=%s driver=%s',
                     version.api_version.split(b'\0', 1)[0].decode(errors='replace'),
                     version.drv_version.split(b'\0', 1)[0].decode(errors='replace'))

            io_num = RknnInputOutputNum()
            self._query(self.RKNN_QUERY_IN_OUT_NUM, io_num)
            self._num_inputs = io_num.n_input
            self._num_outputs = io_num.n_output
            log.info('RKNN model tensors: inputs=%d outputs=%d', self._num_inputs, self._num_outputs)
            if self._num_inputs != 1 or self._num_outputs != 2:
                raise RuntimeError('unsupported model I/O count')

            for index in range(self._num_inputs):
                attr = RknnTensorAttr(index=index)
                self._query(self.RKNN_QUERY_INPUT_ATTR, attr)
                log.info('RKNN input: %s', self._tensor_desc(attr))
                if index == 0 and RKNN_INPUT_MODE == 'uint8':
                    self._input_type = self.RKNN_TENSOR_UINT8
            for index in range(self._num_outputs):
                attr = RknnTensorAttr(index=index)
                self._query(self.RKNN_QUERY_OUTPUT_ATTR, attr)
                log.info('RKNN output: %s', self._tensor_desc(attr))
            log.info('RKNN external input mode: %s', RKNN_INPUT_MODE)
        except Exception as exc:
            raise RuntimeError(f'failed to query RKNN model metadata: {exc}') from exc

    def infer(self, img_rgb: np.ndarray) -> tuple[list, Dict[str, float]]:
        """
        输入: RGB uint8 numpy array shape=(416,416,3)
        输出: list of numpy arrays (每个输出 tensor, float32)

        外部输入为 RGB NHWC；默认 UINT8 由 RKNN 按模型量化参数转换。
        """
        assert img_rgb.shape == (INFER_HEIGHT, INFER_WIDTH, 3)
        assert img_rgb.dtype == np.uint8

        timings = {}
        prepare_start = time.perf_counter()
        if RKNN_INPUT_MODE == 'uint8':
            inp_data = np.ascontiguousarray(img_rgb)
        elif RKNN_INPUT_MODE == 'fp32':
            inp_data = np.ascontiguousarray(img_rgb.astype(np.float32))
        else:
            raise ValueError(f'Unsupported RKNN_INPUT_MODE: {RKNN_INPUT_MODE}')
        inp = RknnInput(
            index=0,
            buf=inp_data.ctypes.data_as(ctypes.c_void_p),
            size=inp_data.nbytes,
            pass_through=0,
            type=self._input_type,
            fmt=self.RKNN_TENSOR_NHWC,
        )
        ret = self._lib.rknn_inputs_set(self._ctx, 1, ctypes.byref(inp))
        if ret != self.RKNN_SUCC:
            raise RuntimeError(f'rknn_inputs_set failed: {ret}')
        timings['prepare_ms'] = (time.perf_counter() - prepare_start) * 1000

        # ── 推理 ────────────────────────────────────────────
        run_start = time.perf_counter()
        ret = self._lib.rknn_run(self._ctx, None)
        if ret != self.RKNN_SUCC:
            raise RuntimeError(f'rknn_run failed: {ret}')
        timings['run_call_ms'] = (time.perf_counter() - run_start) * 1000

        # ── 获取输出 (ppyoloe_split: 2 个输出 tensor) ────────
        output_start = time.perf_counter()
        outputs = (RknnOutput * self._num_outputs)()
        for index, o in enumerate(outputs):
            o.want_float = 1
            o.index = index

        ret = self._lib.rknn_outputs_get(self._ctx, self._num_outputs, outputs, None)
        if ret != self.RKNN_SUCC:
            raise RuntimeError(f'rknn_outputs_get failed: {ret}')

        results = []
        try:
            for o in outputs:
                if o.buf and o.size:
                    arr = np.frombuffer(
                        (ctypes.c_uint8 * o.size).from_address(o.buf),
                        dtype=np.float32
                    ).copy()
                    results.append(arr)
        finally:
            self._lib.rknn_outputs_release(self._ctx, self._num_outputs, outputs)
        timings['output_ms'] = (time.perf_counter() - output_start) * 1000

        perf = RknnPerfRun()
        ret = self._lib.rknn_query(self._ctx, self.RKNN_QUERY_PERF_RUN,
                                   ctypes.byref(perf), ctypes.sizeof(perf))
        if ret == self.RKNN_SUCC and perf.run_duration > 0:
            timings['npu_ms'] = perf.run_duration / 1000.0
        return results, timings

    def close(self):
        if self._ctx:
            self._lib.rknn_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()


# ── 后处理：ppyoloe_split 输出解析 ────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def decode_ppyoloe(outputs: list, conf_thresh: float, nms_thresh: float,
                   img_w: int, img_h: int) -> List[Detection]:
    """
    ppyoloe_split 实际输出（已通过推理确认）:
            outputs[0]: boxes  [3549, 4]  绝对像素坐标 (x1, y1, x2, y2) 相对于416×416 stretch空间
      outputs[1]: scores [3549, 10] 已经是概率值，无需 sigmoid

        输出 bbox 坐标: SOURCE_WIDTH×SOURCE_HEIGHT 源视频空间
    """
    if len(outputs) < 2:
        return []

    num_classes = len(CLASS_NAMES)  # 10

    # output[0]: boxes, output[1]: scores（根据实际推理确认）
    boxes_raw  = outputs[0]   # n_floats=14196 → [3549,4]
    scores_raw = outputs[1]   # n_floats=35490 → [3549,10]

    num_boxes = boxes_raw.size // 4
    boxes  = boxes_raw.reshape(num_boxes, 4)
    scores = scores_raw.reshape(num_boxes, num_classes)
    # 分数已是概率（0~1），不需要 sigmoid

    best_conf = np.max(scores, axis=1)
    best_cls = np.argmax(scores, axis=1)
    mask = best_conf >= conf_thresh
    if not mask.any():
        return []

    sc = best_conf[mask]
    cl = best_cls[mask]
    bx = boxes[mask]

    x1 = np.clip(bx[:, 0] * _SCALE_X, 0, img_w).astype(int)
    y1 = np.clip(bx[:, 1] * _SCALE_Y, 0, img_h).astype(int)
    x2 = np.clip(bx[:, 2] * _SCALE_X, 0, img_w).astype(int)
    y2 = np.clip(bx[:, 3] * _SCALE_Y, 0, img_h).astype(int)

    valid = (x2 > x1) & (y2 > y1)
    if not valid.any():
        return []
    x1, y1, x2, y2, sc, cl = x1[valid], y1[valid], x2[valid], y2[valid], sc[valid], cl[valid]

    boxes_xywh = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])]
                  for i in range(len(x1))]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, sc.tolist(), conf_thresh, nms_thresh)
    flat = np.array(indices).flatten() if len(indices) > 0 else []

    detections = []
    for i in flat:
        cls_id = int(cl[i])
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
        detections.append(Detection(len(detections), class_name,
                                    [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                                    round(float(sc[i]), 4)))

    return sorted(detections, key=lambda d: -d.confidence)


def nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    if not dets:
        return []
    boxes  = np.array([d.bbox for d in dets], dtype=np.float32)
    scores = np.array([d.confidence for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return [dets[k] for k in keep]


# ── 帧采集器（GStreamer appsink，取 raw RGB） ─────────────────────────────

class FrameGrabber:
    """持续保存硬解并缩放后的最新 416×416 RGB 帧。"""

    def __init__(self):
        Gst.init(None)
        if VIDEO_SOURCE == 'shm':
            source = (
                f'shmsrc socket-path={DETECT_SHM_SOCKET} is-live=true do-timestamp=false '
            )
        else:
            source = (
                f'v4l2src device={VIDEO_DEVICE} '
                f'! image/jpeg,width={SOURCE_WIDTH},height={SOURCE_HEIGHT},framerate=30/1 '
            )
        pipeline_str = (
            source +
            f'! image/jpeg,width={SOURCE_WIDTH},height={SOURCE_HEIGHT},framerate=30/1 '
            '! jpegparse '
            '! mppjpegdec format=NV12 fast-mode=true ignore-error=true '
            f'! videoscale ! video/x-raw,format=NV12,width={INFER_WIDTH},height={INFER_HEIGHT} '
            '! videoconvert '
            f'! video/x-raw,format=RGB,width={INFER_WIDTH},height={INFER_HEIGHT} '
            '! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false'
        )
        self._frame_lock = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._sequence = 0
        self._latest_pts_ns = 0
        self._latest_capture_ts = 0.0
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._sink = self._pipeline.get_by_name('sink')
        self._sink.connect('new-sample', self._on_new_sample)
        self._pipeline.set_state(Gst.State.PLAYING)
        log.info('FrameGrabber pipeline started  →  %dx%d RGB', INFER_WIDTH, INFER_HEIGHT)

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(INFER_HEIGHT, INFER_WIDTH, 3).copy()
        pts_ns = int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else 0
        buf.unmap(mapinfo)
        with self._frame_lock:
            self._latest_frame = frame
            self._sequence += 1
            self._latest_pts_ns = pts_ns
            self._latest_capture_ts = time.time()
            self._frame_lock.notify_all()
        return Gst.FlowReturn.OK

    def grab_latest(self, last_sequence: int, timeout: float = 0.1) -> tuple[int, int, float, np.ndarray | None]:
        """等待下一张相机帧；只返回缓存中的最新帧，丢弃过时帧。"""
        deadline = time.monotonic() + timeout
        with self._frame_lock:
            while self._sequence == last_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return last_sequence, 0, 0.0, None
                self._frame_lock.wait(remaining)
            return self._sequence, self._latest_pts_ns, self._latest_capture_ts, self._latest_frame

    def __del__(self):
        if hasattr(self, '_pipeline') and self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)


# ── WebSocket 服务 ────────────────────────────────────────────────────────

class DetectionServer:
    def __init__(self):
        self._clients: set = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._grabber: FrameGrabber | None = None
        self._rknn: RKNNRuntime | None = None
        self._running = False
        self._infer_thread: threading.Thread | None = None
        self._infer_count = 0
        self._timing_sums: Dict[str, float] = {}
        self._last_stats_ts = time.time()
        self._send_pending = False

    def set_event_loop(self, loop):
        self._loop = loop

    def add_client(self, ws):
        with self._lock:
            self._clients.add(ws)
        log.info('client connected, total=%d', len(self._clients))

    def remove_client(self, ws):
        with self._lock:
            self._clients.discard(ws)

    def start(self):
        log.info('Loading RKNN model: %s', RKNN_MODEL)
        self._rknn = RKNNRuntime(RKNN_MODEL)
        self._grabber = FrameGrabber()
        self._running = True
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()
        log.info('Detection WebSocket listening on ws://0.0.0.0:%d', DETECTION_PORT)

    def _infer_loop(self):
        target_interval = 1.0 / max(DETECTION_FPS, 1.0)
        last_sequence = 0
        while self._running:
            wait_start = time.perf_counter()
            sequence, pts_ns, capture_ts, frame = self._grabber.grab_latest(last_sequence)
            wait_ms = (time.perf_counter() - wait_start) * 1000
            if frame is None:
                continue
            last_sequence = sequence

            infer_start = time.perf_counter()
            try:
                outputs, timings = self._rknn.infer(frame)
                post_start = time.perf_counter()
                dets = decode_ppyoloe(outputs, CONF_THRESH, NMS_THRESH,
                                       SOURCE_WIDTH, SOURCE_HEIGHT)
                # 只保留置信度最高的一个检测结果
                dets = dets[:1]
                timings['post_ms'] = (time.perf_counter() - post_start) * 1000
            except Exception as e:
                log.warning('Inference error: %s', e)
                dets = []
                timings = {}

            latency_ms = (time.perf_counter() - infer_start) * 1000
            timings['wait_ms'] = wait_ms
            timings['infer_total_ms'] = latency_ms
            self._infer_count += 1
            for key, value in timings.items():
                self._timing_sums[key] = self._timing_sums.get(key, 0.0) + value
            self._maybe_log_stats()

            payload = self._build_payload(dets, latency_ms, sequence, pts_ns, capture_ts)
            msg = json.dumps(payload)

            if dets:
                top = dets[0]
                log.info(
                    'hit frame_id=%d pts_ns=%d class=%s conf=%.3f bbox=%s',
                    sequence,
                    pts_ns,
                    top.class_name,
                    top.confidence,
                    top.bbox,
                )

            if self._loop and self._clients and not self._send_pending:
                with self._lock:
                    clients = list(self._clients)
                self._send_pending = True
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_and_clear(msg, clients), self._loop
                )

            elapsed = time.perf_counter() - infer_start
            sleep = target_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _maybe_log_stats(self):
        now = time.time()
        elapsed = now - self._last_stats_ts
        if elapsed < 5.0:
            return
        count = max(self._infer_count, 1)
        avg = lambda key: self._timing_sums.get(key, 0.0) / count
        log.info(
            'stats infer_fps=%.1f target_fps=%.1f wait=%.1fms prepare=%.1fms '
            'run=%.1fms npu=%.1fms output=%.1fms post=%.1fms total=%.1fms clients=%d',
            self._infer_count / elapsed, DETECTION_FPS, avg('wait_ms'), avg('prepare_ms'),
            avg('run_call_ms'), avg('npu_ms'), avg('output_ms'), avg('post_ms'),
            avg('infer_total_ms'), len(self._clients))
        self._infer_count = 0
        self._timing_sums.clear()
        self._last_stats_ts = now

    @staticmethod
    def _build_payload(dets: List[Detection], latency_ms: float,
                       frame_id: int, frame_pts_ns: int, capture_ts: float) -> dict:
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        cap_sec = int(capture_ts) if capture_ts > 0 else 0
        cap_nanosec = int((capture_ts - cap_sec) * 1e9) if capture_ts > 0 else 0
        return {
            'type':       'detection',
            'stamp':      {'sec': sec, 'nanosec': nanosec},
            'frame_id':   frame_id,
            'frame_pts_ns': frame_pts_ns,
            'capture_stamp': {'sec': cap_sec, 'nanosec': cap_nanosec},
            'latency_ms': round(latency_ms, 2),
            'input_size': [SOURCE_WIDTH, SOURCE_HEIGHT],
            'detections': [
                {
                    'id':         d.id,
                    'class_name': d.class_name,
                    'bbox':       d.bbox,
                    'confidence': d.confidence,
                }
                for d in dets
            ]
        }

    async def _broadcast(self, msg: str, clients: list):
        await asyncio.gather(
            *[self._safe_send(ws, msg) for ws in clients],
            return_exceptions=True
        )

    async def _broadcast_and_clear(self, msg: str, clients: list):
        try:
            await self._broadcast(msg, clients)
        finally:
            self._send_pending = False

    @staticmethod
    async def _safe_send(ws, msg: str):
        await ws.send(msg)

    def stop(self):
        self._running = False
        if self._rknn:
            self._rknn.close()


server = DetectionServer()


async def handler(websocket):
    server.add_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        server.remove_client(websocket)


async def serve():
    server.set_event_loop(asyncio.get_running_loop())
    server.start()
    async with websockets.serve(handler, '0.0.0.0', DETECTION_PORT):
        await asyncio.Future()


def main():
    asyncio.run(serve())


if __name__ == '__main__':
    main()
