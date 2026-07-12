"""
视频流 WebSocket 服务
- 从 shmsrc 读取原始 MJPEG 帧（相机直出，无需解码再编码）
- 每条 binary message = 24字节帧头 + 完整 JPEG 字节
- 帧头格式: magic='VJPG'(4B) + frame_id(uint64) + pts_ns(uint64) + jpeg_size(uint32)
- 前端按 frame_id/pts 与检测结果精确对齐后再渲染
"""

import asyncio
import struct
import logging
import os
import threading
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import websockets

logging.basicConfig(level=logging.INFO, format='[video] %(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('video_server')

# ── 配置 ──────────────────────────────────────────────────────────────────
VIDEO_DEVICE  = os.environ.get('VIDEO_DEVICE',  '/dev/video20')
VIDEO_SOURCE  = os.environ.get('VIDEO_SOURCE',  'shm')
VIDEO_SHM_SOCKET = os.environ.get('VIDEO_SHM_SOCKET', '/tmp/camera_video.shm')
WIDTH         = int(os.environ.get('WIDTH',     '1920'))
HEIGHT        = int(os.environ.get('HEIGHT',    '1080'))
FRAMERATE     = int(os.environ.get('FRAMERATE', '30'))
WS_PORT       = int(os.environ.get('WS_PORT',   '8765'))


class VideoStreamer:
    """GStreamer 管道：MJPEG 直通 → appsink。"""

    def __init__(self):
        Gst.init(None)
        self._clients: set = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pipeline: Gst.Pipeline | None = None
        self._pending_send = None
        self._sample_count = 0
        self._sent_count = 0
        self._drop_count = 0
        self._last_stats_ts = time.time()
        self._jpeg_probe_logged = False
        self._frame_id = 0

    MAGIC = b'VJPG'
    HEADER_STRUCT = struct.Struct('!4sQQI')

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def add_client(self, ws):
        with self._lock:
            self._clients.add(ws)
        log.info('client connected, total=%d', len(self._clients))

    def remove_client(self, ws):
        with self._lock:
            self._clients.discard(ws)
        log.info('client disconnected, total=%d', len(self._clients))

    # ── GStreamer 管道 ──────────────────────────────────────────────────
    def _build_pipeline(self) -> str:
        if VIDEO_SOURCE == 'shm':
            source = (
                f'shmsrc socket-path={VIDEO_SHM_SOCKET} is-live=true do-timestamp=false '
            )
        else:
            source = (
                f'v4l2src device={VIDEO_DEVICE} '
                f'! image/jpeg,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/1 '
            )
        # 直接转发 JPEG，不做任何解码/编码，延迟最低
        return (
            source +
            f'! image/jpeg,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/1 '
            '! jpegparse '
            '! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false'
        )

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR

        jpeg = bytes(mapinfo.data)
        pts_ns = int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else 0
        buf.unmap(mapinfo)
        self._frame_id += 1
        header = self.HEADER_STRUCT.pack(self.MAGIC, self._frame_id, pts_ns, len(jpeg))
        data = header + jpeg

        if not self._jpeg_probe_logged:
            is_jpeg = len(jpeg) >= 4 and jpeg[:2] == b'\xff\xd8' and jpeg[-2:] == b'\xff\xd9'
            log.info(
                'sample probe frame_id=%d pts_ns=%d jpeg_bytes=%d soi=%s eoi=%s head=%s tail=%s',
                self._frame_id,
                pts_ns,
                len(jpeg),
                jpeg[:2].hex() if len(jpeg) >= 2 else '',
                jpeg[-2:].hex() if len(jpeg) >= 2 else '',
                jpeg[:8].hex(),
                jpeg[-8:].hex(),
            )
            if not is_jpeg:
                log.error('appsink sample is not a complete JPEG frame')
            self._jpeg_probe_logged = True

        # 异步推送给所有客户端
        self._sample_count += 1
        if self._loop and self._clients:
            if self._pending_send and not self._pending_send.done():
                self._drop_count += 1
                self._maybe_log_stats()
                return Gst.FlowReturn.OK
            with self._lock:
                clients = list(self._clients)
            self._pending_send = asyncio.run_coroutine_threadsafe(
                self._broadcast(data, clients), self._loop
            )
            self._sent_count += 1

        self._maybe_log_stats()

        return Gst.FlowReturn.OK

    def _maybe_log_stats(self):
        now = time.time()
        elapsed = now - self._last_stats_ts
        if elapsed < 5.0:
            return
        log.info('stats input_fps=%.1f sent_fps=%.1f dropped=%d clients=%d',
                 self._sample_count / elapsed,
                 self._sent_count / elapsed,
                 self._drop_count,
                 len(self._clients))
        self._sample_count = 0
        self._sent_count = 0
        self._drop_count = 0
        self._last_stats_ts = now

    async def _broadcast(self, data: bytes, clients: list):
        if not clients:
            return
        results = await asyncio.gather(
            *[self._safe_send(ws, data) for ws in clients],
            return_exceptions=True
        )
        for ws, r in zip(clients, results):
            if isinstance(r, Exception):
                log.debug('send error, dropping client: %s', r)
                self.remove_client(ws)

    @staticmethod
    async def _safe_send(ws, data: bytes):
        await ws.send(data)

    def start(self):
        pipeline_str = self._build_pipeline()
        log.info('Pipeline: %s', pipeline_str)
        self._pipeline = Gst.parse_launch(pipeline_str)

        sink = self._pipeline.get_by_name('sink')
        sink.connect('new-sample', self._on_new_sample)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Failed to start GStreamer pipeline')
        log.info('GStreamer pipeline PLAYING  device=%s  %dx%d@%dfps  mode=MJPEG-passthrough',
                 VIDEO_DEVICE, WIDTH, HEIGHT, FRAMERATE)

        # GLib main loop（处理 GStreamer bus 消息）
        glib_loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::error', self._on_error)
        bus.connect('message::eos',   self._on_eos)
        threading.Thread(target=glib_loop.run, daemon=True).start()

    def stop(self):
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            log.info('Pipeline stopped')

    def _on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        log.error('GStreamer error: %s  debug: %s', err, dbg)

    def _on_eos(self, bus, msg):
        log.warning('GStreamer EOS received')


# ── WebSocket 处理 ────────────────────────────────────────────────────────

streamer = VideoStreamer()


async def handler(websocket):
    streamer.add_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        streamer.remove_client(websocket)


async def serve():
    streamer.set_event_loop(asyncio.get_running_loop())
    streamer.start()
    log.info('Video WebSocket listening on ws://0.0.0.0:%d', WS_PORT)
    async with websockets.serve(handler, '0.0.0.0', WS_PORT, compression=None):
        await asyncio.Future()  # run forever


def main():
    asyncio.run(serve())


if __name__ == '__main__':
    main()
