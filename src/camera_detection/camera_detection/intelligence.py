"""Deterministic scene understanding and local text generation."""

from dataclasses import asdict, dataclass
import json
import time
import urllib.request
import uuid


def bbox_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: list
    confidence: float
    first_seen_ms: int
    last_seen_ms: int
    hits: int = 1
    missed: int = 0
    confirmed: bool = False

    def payload(self) -> dict:
        return asdict(self)


class IoUTracker:
    """Small dependency-free tracker; replaceable with ByteTrack later."""

    def __init__(self, iou_threshold=0.3, confirm_hits=3, max_missed=6):
        self.iou_threshold = iou_threshold
        self.confirm_hits = confirm_hits
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks = {}

    def update(self, detections, now_ms=None):
        now_ms = now_ms or round(time.time() * 1000)
        unmatched = set(self._tracks)
        results = []
        for detection in sorted(detections, key=lambda item: -item.confidence):
            candidates = [
                (bbox_iou(track.bbox, detection.bbox), track_id)
                for track_id, track in self._tracks.items()
                if track_id in unmatched and track.class_name == detection.class_name
            ]
            score, track_id = max(candidates, default=(0.0, None))
            if track_id is None or score < self.iou_threshold:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(
                    track_id, detection.class_name, list(detection.bbox),
                    detection.confidence, now_ms, now_ms)
            else:
                track = self._tracks[track_id]
                unmatched.discard(track_id)
                track.bbox = list(detection.bbox)
                track.confidence = round(
                    track.confidence * 0.6 + detection.confidence * 0.4, 4)
                track.last_seen_ms = now_ms
                track.hits += 1
                track.missed = 0
                track.confirmed = track.hits >= self.confirm_hits
            results.append(self._tracks[track_id])

        for track_id in list(unmatched):
            track = self._tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                del self._tracks[track_id]
        return results

    def active(self):
        return [track for track in self._tracks.values() if track.confirmed]


class SceneEngine:
    def __init__(self):
        self.revision = 0
        self._active_ids = set()

    def update(self, tracks, frame_id, telemetry=None, now_ms=None):
        now_ms = now_ms or round(time.time() * 1000)
        active = [track for track in tracks if track.confirmed]
        active_ids = {track.track_id for track in active}
        entered = active_ids - self._active_ids
        left = self._active_ids - active_ids
        events = []
        for track in active:
            if track.track_id in entered:
                events.append(self._event('object_entered', track, now_ms))
        for track_id in left:
            events.append(self._event('object_left', None, now_ms, track_id))
        self._active_ids = active_ids
        self.revision += 1
        primary = max(active, key=lambda item: item.confidence, default=None)
        scene = {
            'type': 'scene_snapshot', 'version': 1, 'revision': self.revision,
            'stamp_ms': now_ms, 'frame_id': frame_id,
            'primary_track_id': primary.track_id if primary else None,
            'tracks': [track.payload() for track in active],
            'telemetry': telemetry or {},
        }
        return scene, events

    def _event(self, kind, track, now_ms, track_id=None):
        resolved_id = track.track_id if track else track_id
        return {
            'type': 'event', 'version': 1,
            'event_id': uuid.uuid4().hex, 'event_type': kind,
            'track_id': resolved_id, 'class_name': track.class_name if track else None,
            'severity': 'info', 'stamp_ms': now_ms,
        }


class TemplateGenerator:
    def generate(self, event, scene_revision):
        class_name = event.get('class_name') or '目标'
        if event['event_type'] == 'object_entered':
            text = f'检测到{class_name}'
        elif event['event_type'] == 'object_left':
            text = '目标已离开视野'
        else:
            text = '场景状态已更新'
        now_ms = round(time.time() * 1000)
        return {
            'type': 'description', 'version': 1,
            'event_id': event['event_id'], 'track_id': event.get('track_id'),
            'scene_revision': scene_revision, 'source': 'template',
            'text': text, 'generated_at': now_ms, 'expires_at': now_ms + 5000,
            'severity': event.get('severity', 'info'),
        }


class CloudGenerator:
    """OpenAI-compatible event generator with cache and circuit breaker."""

    def __init__(self, url, api_key, model, timeout=2.5, failure_limit=3,
                 cooldown=30.0):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.failure_limit = failure_limit
        self.cooldown = cooldown
        self._failures = 0
        self._open_until = 0.0
        self._cache = {}

    @property
    def enabled(self):
        return bool(self.url and self.api_key and self.model)

    def generate(self, event, scene):
        if not self.enabled or time.monotonic() < self._open_until:
            return None
        cache_key = (event.get('event_type'), event.get('class_name'))
        if cache_key in self._cache:
            text = self._cache[cache_key]
        else:
            try:
                text = self._request(event, scene)
                self._cache[cache_key] = text
                self._failures = 0
            except Exception:
                self._failures += 1
                if self._failures >= self.failure_limit:
                    self._open_until = time.monotonic() + self.cooldown
                raise
        now_ms = round(time.time() * 1000)
        return {
            'type': 'description', 'version': 1,
            'event_id': event['event_id'], 'track_id': event.get('track_id'),
            'scene_revision': scene['revision'], 'source': 'cloud',
            'text': text, 'generated_at': now_ms, 'expires_at': now_ms + 5000,
            'severity': event.get('severity', 'info'),
        }

    def _request(self, event, scene):
        context = {
            'event_type': event.get('event_type'),
            'class_name': event.get('class_name'),
            'track_id': event.get('track_id'),
            'telemetry': scene.get('telemetry', {}),
        }
        prompt = (
            '你是智能车HUD播报器。根据JSON事件生成一句不超过30个汉字的客观中文提示，'
            '不要添加JSON中不存在的信息。\n' +
            json.dumps(context, ensure_ascii=False))
        body = json.dumps({
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 80, 'stream': False,
        }).encode('utf-8')
        request = urllib.request.Request(
            self.url + '/chat/completions', data=body,
            headers={'Authorization': f'Bearer {self.api_key}',
                     'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read())
        text = data['choices'][0]['message']['content'].strip()
        if not text or len(text) > 60:
            raise ValueError('cloud description is empty or too long')
        return text