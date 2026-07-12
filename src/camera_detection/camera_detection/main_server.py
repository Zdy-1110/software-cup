"""
统一服务：一路摄像头 → GStreamer tee 分流
  分支1: jpegdec → mpph264enc → appsink_video → WebSocket 8765
  分支2: jpegdec → videoscale 416×416 RGB → appsink_infer → RKNN → WebSocket 8766
"""

import asyncio
import concurrent.futures
import ctypes
import json
import logging
import os
import queue
import threading
import time
import urllib.request
from collections import namedtuple
from typing import List

import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='[main] %(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger('main_server')

# ── 配置 ──────────────────────────────────────────────────────────────────
VIDEO_DEVICE   = os.environ.get('VIDEO_DEVICE',  '/dev/video20')
WIDTH          = int(os.environ.get('WIDTH',     '1920'))
HEIGHT         = int(os.environ.get('HEIGHT',    '1080'))
FRAMERATE      = int(os.environ.get('FRAMERATE', '30'))
BITRATE_KBPS   = int(os.environ.get('H264_BITRATE', '8000'))
GOP            = int(os.environ.get('H264_GOP',  '8'))
WS_PORT        = int(os.environ.get('WS_PORT',   '8765'))
DETECTION_PORT = int(os.environ.get('DETECTION_PORT', '8766'))
RKNN_MODEL     = os.environ.get('RKNN_MODEL',
                 '/home/teamhd/Downloads/ppyoloe_carrace_rk3588_official_split_int8_416.rknn')
INFER_W        = 416
INFER_H        = 416
CONF_THRESH    = float(os.environ.get('CONF_THRESH', '0.3'))

# ── Letterbox 参数（从相机分辨率到推理尺寸，保持宽高比）────────────────────
# 1920×1080 → 416×416: scale=416/1920≈0.2167, pad_x=0, pad_y≈91
_LB_SCALE = min(INFER_W / 1920, INFER_H / 1080)   # 用字面量，WIDTH/HEIGHT 在此行前已定义
_LB_SCALED_W = round(1920 * _LB_SCALE)             # 416
_LB_SCALED_H = round(1080 * _LB_SCALE)             # 234
_LB_PAD_X    = (INFER_W - _LB_SCALED_W) / 2        # 0.0
_LB_PAD_Y    = (INFER_H - _LB_SCALED_H) / 2        # 91.0
NMS_THRESH     = float(os.environ.get('NMS_THRESH',  '0.45'))
CLASS_NAMES    = os.environ.get(
    'CLASS_NAMES', 'bm,cjl,jsjd,jzt,lu,mtl,nc,tt,ydm,zynsx'
).split(',')

# ── 云端 API 配置（OpenAI 兼容，留空则关闭）────────────────────────────────
# 阿里云示例：CLOUD_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# OpenAI 示例：CLOUD_API_URL=https://api.openai.com/v1
CLOUD_API_URL      = os.environ.get('CLOUD_API_URL',   '')
CLOUD_API_KEY      = os.environ.get('CLOUD_API_KEY',   '')
CLOUD_API_MODEL    = os.environ.get('CLOUD_API_MODEL', 'qwen-turbo')
CLOUD_API_DEBOUNCE = float(os.environ.get('CLOUD_API_DEBOUNCE', '30'))  # 同一类别再次触发的最短间隔(秒)
CLOUD_API_PROMPT   = os.environ.get(
    'CLOUD_API_PROMPT',
    '你是一个世界地标与动物百科助手。沙盘中识别到了微缩模型：{class_name}。用2-3句话简洁有趣地介绍它的真实原型（地理位置、特点或有趣冷知识）。回答要简短口语化，适合快速阅读。'
)

Detection = namedtuple('Detection', ['id', 'class_name', 'bbox', 'confidence'])


# ── RKNN Runtime ──────────────────────────────────────────────────────────

class RKNNRuntime:
    def __init__(self, model_path: str):
        self._lib = ctypes.CDLL('/usr/lib/librknnrt.so')
        self._setup_prototypes()
        with open(model_path, 'rb') as f:
            buf = ctypes.create_string_buffer(f.read())
        ctx = ctypes.c_void_p()
        ret = self._lib.rknn_init(ctypes.byref(ctx), buf, len(buf) - 1, 0, None)
        if ret != 0:
            raise RuntimeError(f'rknn_init failed: {ret}')
        self._ctx = ctx
        log.info('RKNN model loaded: %s', model_path)

    def _setup_prototypes(self):
        L = self._lib
        vp = ctypes.c_void_p
        u32 = ctypes.c_uint32
        ci  = ctypes.c_int
        L.rknn_init.restype           = ci
        L.rknn_init.argtypes          = [ctypes.POINTER(vp), vp, u32, u32, vp]
        L.rknn_inputs_set.restype     = ci
        L.rknn_inputs_set.argtypes    = [vp, u32, vp]
        L.rknn_run.restype            = ci
        L.rknn_run.argtypes           = [vp, vp]
        L.rknn_outputs_get.restype    = ci
        L.rknn_outputs_get.argtypes   = [vp, u32, vp, vp]
        L.rknn_outputs_release.restype  = ci
        L.rknn_outputs_release.argtypes = [vp, u32, vp]
        L.rknn_destroy.restype        = ci
        L.rknn_destroy.argtypes       = [vp]

    def infer(self, img_rgb: np.ndarray) -> list:
        assert img_rgb.shape == (INFER_H, INFER_W, 3) and img_rgb.dtype == np.uint8

        class RknnInput(ctypes.Structure):
            _fields_ = [('index', ctypes.c_uint32), ('buf', ctypes.c_void_p),
                        ('size', ctypes.c_size_t), ('pass_through', ctypes.c_uint8),
                        ('type', ctypes.c_int32), ('fmt', ctypes.c_int32)]

        # FP32 NHWC [0,255] — 训练照片实测 max_score=0.88（不除255）
        data = np.ascontiguousarray(img_rgb.astype(np.float32))
        inp  = RknnInput(index=0, buf=data.ctypes.data_as(ctypes.c_void_p),
                         size=data.nbytes, pass_through=0, type=1, fmt=1)
        if self._lib.rknn_inputs_set(self._ctx, 1, ctypes.byref(inp)) != 0:
            raise RuntimeError('rknn_inputs_set failed')
        if self._lib.rknn_run(self._ctx, None) != 0:
            raise RuntimeError('rknn_run failed')

        class RknnOutput(ctypes.Structure):
            _fields_ = [('want_float', ctypes.c_uint8), ('is_prealloc', ctypes.c_uint8),
                        ('_pad', ctypes.c_uint8 * 6), ('buf', ctypes.c_void_p),
                        ('size', ctypes.c_uint32)]

        NUM_OUT = 2
        outs = (RknnOutput * NUM_OUT)()
        for o in outs:
            o.want_float = 1
        if self._lib.rknn_outputs_get(self._ctx, NUM_OUT, outs, None) != 0:
            raise RuntimeError('rknn_outputs_get failed')

        results = []
        for o in outs:
            if o.buf and o.size:
                arr = np.frombuffer(
                    (ctypes.c_uint8 * o.size).from_address(o.buf), dtype=np.float32
                ).copy()
                results.append(arr)
        self._lib.rknn_outputs_release(self._ctx, NUM_OUT, outs)
        return results

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.rknn_destroy(self._ctx)


# ── 后处理 ────────────────────────────────────────────────────────────────

def decode_ppyoloe(outputs: list) -> List[Detection]:
    """
    实测确认输出顺序:
      output[0]: boxes  [3549,4]  cx,cy,w,h 绝对像素（416×416 letterbox 空间）
      output[1]: scores [3549,10] 直接概率值，无需 sigmoid

    输出 bbox 坐标已反变换回源分辨率（WIDTH×HEIGHT，即 1920×1080）。
    前端应使用 input_size 字段做坐标映射，不再除以 416。
    """
    if len(outputs) < 2:
        return []
    num_classes = len(CLASS_NAMES)
    boxes  = outputs[0].reshape(-1, 4)   # [3549,4]
    scores = outputs[1].reshape(-1, num_classes)  # [3549,10]

    best_conf = np.max(scores, axis=1)
    best_cls  = np.argmax(scores, axis=1)
    mask = best_conf >= CONF_THRESH
    if not mask.any():
        return []

    sc = best_conf[mask]; cl = best_cls[mask]; bx = boxes[mask]

    # output[0] 实为 x1,y1,x2,y2（绝对像素，416×416 letterbox 空间）
    # 不是 cx,cy,w,h —— 由实测 tensor 范围含负值确认
    lx1, ly1, lx2, ly2 = bx[:,0], bx[:,1], bx[:,2], bx[:,3]

    # 反变换 letterbox → 源分辨率（1920×1080）
    # x_src = (x_lb - pad_x) / scale，y 同理；超出范围 clip
    x1 = np.clip((lx1 - _LB_PAD_X) / _LB_SCALE, 0, WIDTH).astype(int)
    y1 = np.clip((ly1 - _LB_PAD_Y) / _LB_SCALE, 0, HEIGHT).astype(int)
    x2 = np.clip((lx2 - _LB_PAD_X) / _LB_SCALE, 0, WIDTH).astype(int)
    y2 = np.clip((ly2 - _LB_PAD_Y) / _LB_SCALE, 0, HEIGHT).astype(int)

    valid = (x2 > x1) & (y2 > y1)
    x1,y1,x2,y2,sc,cl = x1[valid],y1[valid],x2[valid],y2[valid],sc[valid],cl[valid]

    boxes_xywh = [[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])]
                  for i in range(len(x1))]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, sc.tolist(), CONF_THRESH, NMS_THRESH)
    flat = np.array(indices).flatten() if len(indices) > 0 else []
    dets = []
    for i in flat:
        c = int(cl[i])
        name = CLASS_NAMES[c] if c < num_classes else f'cls{c}'
        dets.append(Detection(len(dets), name,
                              [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                              round(float(sc[i]), 4)))
    return sorted(dets, key=lambda d: -d.confidence)




# ── 统一 GStreamer 管道（tee 分流）────────────────────────────────────────

class UnifiedPipeline:
    """
    摄像头 → jpegdec → tee
      tee → videoconvert → NV12 → mpph264enc → h264parse → appsink_video
      tee → videoconvert → RGB  → videoscale → appsink_infer
    """

    def __init__(self):
        Gst.init(None)
        self._pipeline: Gst.Pipeline | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # 视频 WebSocket 客户端
        self._video_clients: set = set()
        self._video_lock = threading.Lock()
        # 检测 WebSocket 客户端
        self._detect_clients: set = set()
        self._detect_lock = threading.Lock()

        self._rknn: RKNNRuntime | None = None
        self._infer_q: queue.Queue = queue.Queue(maxsize=1)
        self._infer_thread: threading.Thread | None = None

        # 视频帧独立发送队列，避免 GStreamer 回调被 WebSocket 阻塞
        self._video_q: queue.Queue = queue.Queue(maxsize=2)
        self._video_send_thread: threading.Thread | None = None

        # 云端 API：debounce 记录 + 独立线程池（不阻塞推理）
        self._last_desc_time: dict = {}          # {class_name: timestamp}
        self._api_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def set_event_loop(self, loop):
        self._loop = loop

    def add_video_client(self, ws):
        with self._video_lock:
            self._video_clients.add(ws)
        log.info('video client +1  total=%d', len(self._video_clients))

    def remove_video_client(self, ws):
        with self._video_lock:
            self._video_clients.discard(ws)

    def add_detect_client(self, ws):
        with self._detect_lock:
            self._detect_clients.add(ws)
        log.info('detect client +1 total=%d', len(self._detect_clients))

    def remove_detect_client(self, ws):
        with self._detect_lock:
            self._detect_clients.discard(ws)

    def start(self):
        self._rknn = RKNNRuntime(RKNN_MODEL)

        if CLOUD_API_URL:
            self._api_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='cloud_api'
            )
            log.info('Cloud API enabled: %s  model=%s  debounce=%.0fs',
                     CLOUD_API_URL, CLOUD_API_MODEL, CLOUD_API_DEBOUNCE)
        else:
            log.info('Cloud API disabled (set CLOUD_API_URL to enable)')

        # 推理工作线程
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

        # 视频发送线程（独立于 GStreamer 回调，避免大帧阻塞管道）
        self._video_send_thread = threading.Thread(target=self._video_send_worker, daemon=True)
        self._video_send_thread.start()

        pipeline_str = (
            f'v4l2src device={VIDEO_DEVICE} '
            f'! image/jpeg,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/1 '
            '! jpegdec '
            '! tee name=t '

            # 分支1: H264 → WebSocket
            # leaky=downstream: 编码慢时丢最旧帧，不反压 tee
            # mpph264enc: 关B帧(bframes=0), 低延迟参数
            f't. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream '
            '! videoconvert '
            f'! video/x-raw,format=NV12,width={WIDTH},height={HEIGHT} '
            f'! mpph264enc bps={BITRATE_KBPS * 1000} bps-min={BITRATE_KBPS * 1000} '
            f'bps-max={BITRATE_KBPS * 1000} rc-mode=cbr gop={GOP} '
            'header-mode=each-idr profile=baseline level=4.1 '
            '! h264parse config-interval=-1 '   # 每个IDR帧前插SPS/PPS，客户端可从任意关键帧恢复
            '! video/x-h264,stream-format=byte-stream,alignment=au '
            # drop=true: Python 来不及取走时主动丢帧，不让编码器等待
            '! appsink name=sink_video emit-signals=true max-buffers=2 drop=true sync=false '

            # 分支2: 等比缩放到 416×_LB_SCALED_H（Python侧再pad到416×416）
            f't. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream '
            '! videoconvert '
            '! videoscale '
            f'! video/x-raw,format=RGB,width={INFER_W},height={_LB_SCALED_H} '
            '! appsink name=sink_infer emit-signals=true max-buffers=1 drop=true sync=false'
        )
        log.info('Building pipeline...')
        log.info('GStreamer pipeline: %s', pipeline_str)
        self._pipeline = Gst.parse_launch(pipeline_str)

        sink_video = self._pipeline.get_by_name('sink_video')
        sink_infer = self._pipeline.get_by_name('sink_infer')
        sink_video.connect('new-sample', self._on_video_sample)
        sink_infer.connect('new-sample', self._on_infer_sample)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Failed to start GStreamer pipeline')

        log.info('Pipeline PLAYING  device=%s  %dx%d@%dfps  bitrate=%dkbps  gop=%d',
                 VIDEO_DEVICE, WIDTH, HEIGHT, FRAMERATE, BITRATE_KBPS, GOP)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::error', lambda b, m: log.error('GST error: %s', m.parse_error()[0]))
        threading.Thread(target=GLib.MainLoop().run, daemon=True).start()

    # ── 视频帧回调：仅入队，立即返回不阻塞 GStreamer ────────────────────
    def _on_video_sample(self, appsink) -> Gst.FlowReturn:
        sample = appsink.emit('pull-sample')
        if not sample:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        data = bytes(mi.data)
        buf.unmap(mi)

        # 非阻塞入队：队满时丢弃最旧帧保持实时
        if self._video_clients:
            try:
                self._video_q.get_nowait()   # 先清空旧帧
            except queue.Empty:
                pass
            try:
                self._video_q.put_nowait(data)
            except queue.Full:
                pass
        return Gst.FlowReturn.OK

    def _video_send_worker(self):
        """独立线程：从队列取帧，fire-and-forget 发送，不阻塞等待 event loop"""
        while True:
            data = self._video_q.get()
            if not self._loop or not self._video_clients:
                continue
            with self._video_lock:
                clients = list(self._video_clients)
            # fire-and-forget：不等待协程完成，_video_q 的 get_nowait 抢占已保证只发最新帧
            asyncio.run_coroutine_threadsafe(
                self._broadcast_binary(data, clients, self._video_lock, self._video_clients),
                self._loop
            )

    # ── 推理帧回调 ────────────────────────────────────────────────────────
    def _on_infer_sample(self, appsink) -> Gst.FlowReturn:
        sample = appsink.emit('pull-sample')
        if not sample:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        # 等比缩放帧：416×_LB_SCALED_H，需在 Python 侧填充黑边到 416×416
        scaled = np.frombuffer(mi.data, dtype=np.uint8).reshape(_LB_SCALED_H, INFER_W, 3).copy()
        buf.unmap(mi)

        # letterbox padding：上下各 _LB_PAD_Y 行黑边
        pad_top = int(_LB_PAD_Y)
        pad_bot = INFER_H - _LB_SCALED_H - pad_top
        frame = np.pad(scaled, ((pad_top, pad_bot), (0, 0), (0, 0)), mode='constant')

        # 丢弃旧帧，只保留最新帧（非阻塞）
        try:
            self._infer_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._infer_q.put_nowait(frame)
        except queue.Full:
            pass

        return Gst.FlowReturn.OK

    def _infer_worker(self):
        """单一推理线程，避免 RKNN context 并发竞争"""
        while True:
            frame = self._infer_q.get()  # 阻塞等待
            self._run_infer(frame)

    def _run_infer(self, frame: np.ndarray):
        if not self._detect_clients:
            return
        t0 = time.time()
        try:
            outputs = self._rknn.infer(frame)
            dets    = decode_ppyoloe(outputs)
        except Exception as e:
            log.warning('Inference error: %s', e)
            dets = []

        latency_ms = (time.time() - t0) * 1000
        now = time.time()
        payload = json.dumps({
            'type':        'detection',
            'stamp':       {'sec': int(now), 'nanosec': int((now % 1) * 1e9)},
            'latency_ms':  round(latency_ms, 2),
            'input_size':  [WIDTH, HEIGHT],   # bbox 坐标系：源分辨率，前端用此做映射
            'detections':  [
                {'id': d.id, 'class_name': d.class_name,
                 'bbox': d.bbox, 'confidence': d.confidence}
                for d in dets
            ]
        })

        if dets:
            log.debug('detections=%d  latency=%.1fms', len(dets), latency_ms)
            # 云端 API：取最高置信度目标，debounce 后触发
            if self._api_executor and CLOUD_API_URL:
                top = dets[0]
                last = self._last_desc_time.get(top.class_name, 0)
                if now - last > CLOUD_API_DEBOUNCE:
                    self._last_desc_time[top.class_name] = now
                    self._api_executor.submit(self._fetch_and_push_desc, top.class_name)

        if self._loop and self._detect_clients:
            with self._detect_lock:
                clients = list(self._detect_clients)
            asyncio.run_coroutine_threadsafe(
                self._broadcast_text(payload, clients, self._detect_lock,
                                     self._detect_clients), self._loop
            )

    # ── 云端 API ──────────────────────────────────────────────────────────

    def _call_cloud_api(self, class_name: str) -> str:
        """调用 OpenAI 兼容接口，返回物体描述文本（同步，在线程池中执行）"""
        prompt = CLOUD_API_PROMPT.format(class_name=class_name)
        body = json.dumps({
            'model': CLOUD_API_MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 200,
            'stream': False,
        }).encode('utf-8')
        req = urllib.request.Request(
            CLOUD_API_URL.rstrip('/') + '/chat/completions',
            data=body,
            headers={
                'Authorization': f'Bearer {CLOUD_API_KEY}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data['choices'][0]['message']['content'].strip()

    def _fetch_and_push_desc(self, class_name: str):
        """线程池任务：请求API后将描述推送给所有检测客户端"""
        try:
            text = self._call_cloud_api(class_name)
            msg = json.dumps({
                'type':       'description',
                'class_name': class_name,
                'text':       text,
            })
            log.info('API desc [%s]: %s', class_name, text[:60])
            if self._loop and self._detect_clients:
                with self._detect_lock:
                    clients = list(self._detect_clients)
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_text(msg, clients, self._detect_lock,
                                         self._detect_clients),
                    self._loop
                )
        except Exception as e:
            log.warning('Cloud API error [%s]: %s', class_name, e)

    # ── 广播工具 ──────────────────────────────────────────────────────────
    async def _broadcast_binary(self, data, clients, lock, client_set):
        results = await asyncio.gather(
            *[self._safe_send(ws, data) for ws in clients], return_exceptions=True
        )
        for ws, r in zip(clients, results):
            if isinstance(r, Exception):
                with lock:
                    client_set.discard(ws)

    async def _broadcast_text(self, msg, clients, lock, client_set):
        results = await asyncio.gather(
            *[self._safe_send(ws, msg) for ws in clients], return_exceptions=True
        )
        for ws, r in zip(clients, results):
            if isinstance(r, Exception):
                with lock:
                    client_set.discard(ws)

    @staticmethod
    async def _safe_send(ws, data):
        await ws.send(data)

    def stop(self):
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)


# ── WebSocket 服务 ────────────────────────────────────────────────────────

pipeline = UnifiedPipeline()


async def video_handler(websocket):
    pipeline.add_video_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        pipeline.remove_video_client(websocket)


async def detect_handler(websocket):
    pipeline.add_detect_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        pipeline.remove_detect_client(websocket)


async def serve():
    pipeline.set_event_loop(asyncio.get_running_loop())
    pipeline.start()
    log.info('Video  WebSocket: ws://0.0.0.0:%d', WS_PORT)
    log.info('Detect WebSocket: ws://0.0.0.0:%d', DETECTION_PORT)

    async with websockets.serve(video_handler, '0.0.0.0', WS_PORT), \
               websockets.serve(detect_handler, '0.0.0.0', DETECTION_PORT):
        await asyncio.Future()


def main():
    asyncio.run(serve())


if __name__ == '__main__':
    main()
