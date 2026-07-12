"""Deterministic scene understanding and local text generation."""

from dataclasses import asdict, dataclass
import base64
from collections import OrderedDict
import json
import time
import urllib.request
import uuid


LANDMARK_CARDS = {
    'bm': {
        'display_name': '斑马', 'title': '斑马·草原精灵',
        'subtitle': '非洲大草原',
        'body': '斑马的黑白条纹是进化的奇迹，每头斑马的纹路如指纹般独一无二，是大自然最美的艺术。',
        'tags': ['非洲草原', '神奇居民'], 'cta': '扫描线进场',
    },
    'cjl': {
        'display_name': '长颈鹿', 'title': '长颈鹿·高天守望者',
        'subtitle': '非洲稀树草原',
        'body': '地球上最高的陆地动物，2米长的脖子每天只需喝少量的水，却能俯瞰整片草原的风景。',
        'tags': ['草原', '优雅身影'], 'cta': '从上方滑入',
    },
    'jsjd': {
        'display_name': '金沙酒店', 'title': '金沙酒店·城市地标',
        'subtitle': '当代文旅综合体',
        'body': '弧形玻璃幕墙倒映城市天际线，融购物、文化与美食于一体，是现代都市文旅的璀璨坐标。',
        'tags': ['都市文旅', '璀璨坐标'], 'cta': '光扫过建筑轮廓',
    },
    'jzt': {
        'display_name': '吉萨金字塔', 'title': '金字塔·永恒之墓',
        'subtitle': '古埃及·世界奇迹',
        'body': '建于约4500年前，由230万块石头砌成，每块重达2.5吨。它如何建造，至今仍是谜。',
        'tags': ['古埃及', '人类奇迹'], 'cta': '沙尘粒子进场',
    },
    'lu': {
        'display_name': '鹿', 'title': '鹿·森林使者',
        'subtitle': '温带森林生态',
        'body': '鹿角每年脱落后重新生长，是自然界中最快速的骨骼再生奇观，堪称生命力的象征。',
        'tags': ['森林', '柔韧生命'], 'cta': '树叶飘落动效',
    },
    'mtl': {
        'display_name': '摩天轮', 'title': '摩天轮·云端之眼',
        'subtitle': '城市娱乐地标',
        'body': '高耸的摩天轮能把你送到城市上空，让你在云端俯瞰整座城市的灯火与轮廓。',
        'tags': ['城市夜景', '浪漫坐标'], 'cta': '缓慢旋转光圈',
    },
    'nc': {
        'display_name': '鸟巢', 'title': '鸟巢·钢铁筑梦',
        'subtitle': '国家体育场·北京',
        'body': '2008年北京奥运主场馆，9万吨钢材编织成独特的鸟巢造型，承载着中国体育的荣耀与梦想。',
        'tags': ['中国体育', '永恒象征'], 'cta': '钢架网格动效',
    },
    'tt': {
        'display_name': '埃菲尔铁塔', 'title': '铁塔·铁铸浪漫',
        'subtitle': '巴黎·法国地标',
        'body': '建于1889年，原计划仅保留20年后拆除，如今已成为全球最知名的地标，象征浪漫与创造力。',
        'tags': ['巴黎天际线', '优雅铁架'], 'cta': '灯光闪烁动效',
    },
    'ydm': {
        'display_name': '印度门', 'title': '印度门·荣耀之拱',
        'subtitle': '新德里·历史地标',
        'body': '42米高的拱形纪念碑，铭刻着约9万名一战阵亡印度士兵的名字，是南亚大陆的历史丰碑。',
        'tags': ['南亚大陆', '历史丰碑'], 'cta': '拱形光晕扩散',
    },
    'zynsx': {
        'display_name': '自由女神', 'title': '自由女神·自由之光',
        'subtitle': '纽约·美国象征',
        'body': '法国人民赠给美国的礼物，铜像内藏旋转楼梯，火炬顶端可俯瞰纽约湾，象征自由与希望。',
        'tags': ['自由', '希望'], 'cta': '火炬光芒放射',
    },
}


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
    raw_confidence: float
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
                    detection.confidence, detection.confidence, now_ms, now_ms)
            else:
                track = self._tracks[track_id]
                unmatched.discard(track_id)
                track.bbox = list(detection.bbox)
                track.raw_confidence = detection.confidence
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


class ConfidencePolicy:
    """Three-zone confidence policy with deterministic vision fusion."""

    def __init__(self, review_min=0.25, accept_min=0.60):
        if not 0.0 <= review_min < accept_min <= 1.0:
            raise ValueError('confidence thresholds must satisfy 0 <= review < accept <= 1')
        self.review_min = review_min
        self.accept_min = accept_min

    def state(self, confidence):
        if confidence < self.review_min:
            return 'suppressed'
        if confidence < self.accept_min:
            return 'review'
        return 'accepted'

    def fuse(self, candidate, result):
        model_confidence = float(candidate['track_confidence'])
        vision_confidence = float(result['confidence'])
        confirmed = bool(result['confirmed'])
        same_class = result.get('class_name') == candidate['class_name']
        if not confirmed:
            return {
                'decision': 'rejected',
                'effective_class_name': None,
                'effective_confidence': round(
                    min(model_confidence, 1.0 - vision_confidence), 4),
            }
        return {
            'decision': 'accepted' if same_class else 'reclassified',
            'effective_class_name': result['class_name'],
            'effective_confidence': round(
                model_confidence * 0.55 + vision_confidence * 0.45, 4),
        }


class AcceptanceGate:
    """Emit once after a track remains accepted for consecutive updates."""

    def __init__(self, required_hits=3):
        self.required_hits = required_hits
        self._hits = {}
        self._emitted = set()

    def update(self, tracks, policy):
        active_ids = {track.track_id for track in tracks}
        accepted = []
        for track in tracks:
            key = (track.track_id, track.class_name)
            if policy.state(track.confidence) == 'accepted':
                self._hits[key] = self._hits.get(key, 0) + 1
                if self._hits[key] >= self.required_hits and key not in self._emitted:
                    self._emitted.add(key)
                    accepted.append(track)
            else:
                self._hits.pop(key, None)
        self._hits = {
            key: hits for key, hits in self._hits.items()
            if key[0] in active_ids
        }
        self._emitted = {key for key in self._emitted if key[0] in active_ids}
        return accepted


class HudCardStore:
    """Keep currently valid cards for reconnecting clients."""

    def __init__(self, max_cards=3):
        self.max_cards = max_cards
        self._cards = OrderedDict()

    def put(self, card, now_ms=None):
        now_ms = now_ms or round(time.time() * 1000)
        self.valid(now_ms)
        self._cards[card['card_id']] = card
        while len(self._cards) > self.max_cards:
            self._cards.popitem(last=False)

    def valid(self, now_ms=None):
        now_ms = now_ms or round(time.time() * 1000)
        self._cards = OrderedDict(
            (card_id, card) for card_id, card in self._cards.items()
            if card['expires_at'] > now_ms)
        return list(self._cards.values())


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


class HudCardGenerator:
    """Generate a validated HUD card through an OpenAI-compatible text model."""

    def __init__(self, url, api_key, model, timeout=3.0, failure_limit=3,
                 cooldown=30.0):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.failure_limit = failure_limit
        self.cooldown = cooldown
        self._failures = 0
        self._open_until = 0.0

    @property
    def enabled(self):
        return bool(self.url and self.api_key and self.model)

    def generate(self, context):
        if not self.enabled or time.monotonic() < self._open_until:
            return self.template(context)
        try:
            content = self._request(context)
            card = self.build_card(context, content, self.model)
            self._failures = 0
            return card
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_limit:
                self._open_until = time.monotonic() + self.cooldown
            raise

    def _request(self, context):
        trusted = {
            'class_name': context['class_name'],
            'display_name': context['display_name'],
            'confidence': context['confidence'],
            'reference': context.get('reference', {}),
            'telemetry': context.get('telemetry', {}),
        }
        prompt = (
            '你是一位未来文旅AR副驾助手，语气简洁、有趣、充满知识感。'
            '只能依据给定JSON，不得猜测类别缩写或补充未经提供的事实。'
            '严格返回一个JSON对象，不要Markdown，不要任何多余文字，格式为：'
            '{"title":"10字以内","subtitle":"15字以内",'
            '"body":"不超过60字","tags":["标签1","标签2"],'
            '"cta":"行动提示"}。输入：' +
            json.dumps(trusted, ensure_ascii=False))
        body = json.dumps({
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 512, 'temperature': 0.1, 'stream': False,
            'response_format': {'type': 'json_object'},
        }).encode('utf-8')
        request = urllib.request.Request(
            self.url + '/chat/completions', data=body,
            headers={'Authorization': f'Bearer {self.api_key}',
                     'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read())
        content = data['choices'][0]['message']['content'].strip()
        if content.startswith('```'):
            content = content.strip('`').removeprefix('json').strip()
        return self.validate_content(json.loads(content))

    @staticmethod
    def validate_content(content):
        if not isinstance(content, dict):
            raise ValueError('HUD response must be an object')
        required = {'title', 'subtitle', 'body', 'tags', 'cta'}
        if set(content) != required:
            raise ValueError('HUD response has invalid fields')
        title = str(content['title']).strip()
        subtitle = str(content['subtitle']).strip()
        body = str(content['body']).strip()
        cta = str(content['cta']).strip()
        tags = content['tags']
        if (not title or len(title) > 10 or not subtitle or len(subtitle) > 15
            or not body or len(body) > 60 or not cta or len(cta) > 12):
            raise ValueError('HUD text is empty or too long')
        if not isinstance(tags, list) or not 1 <= len(tags) <= 3:
            raise ValueError('HUD tags must contain 1 to 3 items')
        normalized_tags = [str(tag).strip() for tag in tags]
        if any(not tag or len(tag) > 16 for tag in normalized_tags):
            raise ValueError('HUD tag is empty or too long')
        return {
            'title': title, 'subtitle': subtitle, 'body': body,
            'tags': normalized_tags, 'cta': cta,
        }

    @staticmethod
    def build_card(context, content, source):
        now_ms = round(time.time() * 1000)
        return {
            'type': 'hud_card', 'version': 1,
            'card_id': uuid.uuid4().hex,
            'event_id': context.get('event_id'),
            'track_id': context['track_id'],
            'scene_revision': context['scene_revision'],
            'class_name': context['class_name'],
            'display_name': context['display_name'],
            'confidence': round(float(context['confidence']), 4),
            'source': source,
            **content,
            'presentation': {
                'enter_duration_ms': 300,
                'max_visible_cards': 3,
                'cloud_fallback_ms': 800,
            },
            'severity': 'info', 'generated_at': now_ms,
            'expires_at': now_ms + 15000,
        }

    @classmethod
    def template(cls, context):
        reference = context.get('reference') or {}
        display_name = context['display_name']
        content = {
            'title': reference.get('title', display_name),
            'subtitle': reference.get('subtitle', '智能识别目标'),
            'body': reference.get('body', f'检测到{display_name}'),
            'tags': reference.get('tags', ['智能识别']),
            'cta': reference.get('cta', '继续探索'),
        }
        return cls.build_card(context, content, 'template')


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


class VisionUnderstandingClient:
    """OpenAI-compatible visual confirmation for uncertain detections."""

    def __init__(self, url, api_key, model, timeout=3.0, failure_limit=3,
                 cooldown=30.0):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.failure_limit = failure_limit
        self.cooldown = cooldown
        self._failures = 0
        self._open_until = 0.0

    @property
    def enabled(self):
        return bool(self.url and self.api_key and self.model)

    def confirm(self, jpeg, candidate, allowed_classes):
        if not self.enabled or time.monotonic() < self._open_until:
            return None
        try:
            result = self._request(jpeg, candidate, allowed_classes)
            self._failures = 0
            return result
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_limit:
                self._open_until = time.monotonic() + self.cooldown
            raise

    def _request(self, jpeg, candidate, allowed_classes):
        encoded = base64.b64encode(jpeg).decode('ascii')
        prompt = (
            '你是智能车赛道视觉复核器。候选检测结果为：' +
            json.dumps(candidate, ensure_ascii=False) +
            '。只允许从以下类别中确认：' + ','.join(allowed_classes) +
            '。观察候选框附近目标，严格返回JSON对象，不要Markdown：'
            '{"confirmed":true,"class_name":"类别","confidence":0.0,'
            '"reason":"不超过30字"}。无法确认时confirmed为false。')
        body = json.dumps({
            'model': self.model,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {
                    'url': 'data:image/jpeg;base64,' + encoded}},
            ]}],
            'max_tokens': 120,
            'temperature': 0.1,
            'stream': False,
        }).encode('utf-8')
        request = urllib.request.Request(
            self.url + '/chat/completions', data=body,
            headers={'Authorization': f'Bearer {self.api_key}',
                     'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read())
        content = data['choices'][0]['message']['content'].strip()
        if content.startswith('```'):
            content = content.strip('`').removeprefix('json').strip()
        return self.validate_result(json.loads(content), allowed_classes)

    @staticmethod
    def validate_result(result, allowed_classes):
        if not isinstance(result, dict):
            raise ValueError('visual understanding response must be an object')
        class_name = result.get('class_name')
        if class_name not in allowed_classes:
            result['confirmed'] = False
            result['class_name'] = None
        else:
            result['confirmed'] = bool(result.get('confirmed', False))
        result['confidence'] = max(0.0, min(1.0, float(result.get('confidence', 0))))
        result['reason'] = str(result.get('reason', ''))[:60]
        return result