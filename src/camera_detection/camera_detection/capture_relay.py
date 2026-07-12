"""
Camera capture relay
- Opens VIDEO_DEVICE once.
- Publishes the MJPEG stream to independent video and detection consumers
    without fighting over /dev/video20.
"""

import logging
import os
import signal

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logging.basicConfig(level=logging.INFO, format='[relay] %(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('capture_relay')

VIDEO_DEVICE = os.environ.get('VIDEO_DEVICE', '/dev/video20')
WIDTH = int(os.environ.get('WIDTH', '1920'))
HEIGHT = int(os.environ.get('HEIGHT', '1080'))
FRAMERATE = int(os.environ.get('FRAMERATE', '30'))
VIDEO_SHM_SOCKET = os.environ.get('VIDEO_SHM_SOCKET', '/tmp/camera_video.shm')
DETECT_SHM_SOCKET = os.environ.get('DETECT_SHM_SOCKET', '/tmp/camera_detect.shm')
SHM_SIZE = int(os.environ.get('SHM_SIZE', str(64 * 1024 * 1024)))


def cleanup_socket(path: str):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def build_pipeline() -> str:
    return (
        f'v4l2src device={VIDEO_DEVICE} '
        f'! image/jpeg,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/1 '
        '! tee name=t '
        't. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream '
        f'! shmsink socket-path={VIDEO_SHM_SOCKET} shm-size={SHM_SIZE} '
        'wait-for-connection=false sync=false async=false '
        't. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream '
        f'! shmsink socket-path={DETECT_SHM_SOCKET} shm-size={SHM_SIZE} '
        'wait-for-connection=false sync=false async=false'
    )


def main():
    cleanup_socket(VIDEO_SHM_SOCKET)
    cleanup_socket(DETECT_SHM_SOCKET)

    Gst.init(None)
    loop = GLib.MainLoop()
    pipeline_str = build_pipeline()
    log.info('Pipeline: %s', pipeline_str)
    pipeline = Gst.parse_launch(pipeline_str)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_error(_bus, msg):
        err, dbg = msg.parse_error()
        log.error('GStreamer error: %s  debug: %s', err, dbg)
        loop.quit()

    def on_eos(_bus, _msg):
        log.warning('GStreamer EOS received')
        loop.quit()

    bus.connect('message::error', on_error)
    bus.connect('message::eos', on_eos)

    def stop(_signum=None, _frame=None):
        log.info('Stopping relay')
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError('Failed to start GStreamer pipeline')

    log.info('Relay PLAYING  device=%s  %dx%d@%dfps', VIDEO_DEVICE, WIDTH, HEIGHT, FRAMERATE)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        cleanup_socket(VIDEO_SHM_SOCKET)
        cleanup_socket(DETECT_SHM_SOCKET)


if __name__ == '__main__':
    main()
