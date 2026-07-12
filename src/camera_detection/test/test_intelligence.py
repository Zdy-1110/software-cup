import unittest
from collections import namedtuple

from camera_detection.intelligence import (
    CloudGenerator, ConfidencePolicy, HudCardGenerator, IoUTracker, SceneEngine,
    TemplateGenerator, VisionUnderstandingClient)


Detection = namedtuple('Detection', ['id', 'class_name', 'bbox', 'confidence'])


class IntelligenceTest(unittest.TestCase):
    def setUp(self):
        self.tracker = IoUTracker(confirm_hits=3, max_missed=2)
        self.engine = SceneEngine()
        self.detection = Detection(0, 'bm', [10, 10, 110, 110], 0.9)

    def test_track_confirmation_and_stable_id(self):
        ids = []
        for frame in range(3):
            tracks = self.tracker.update([self.detection], 1000 + frame)
            ids.append(tracks[0].track_id)
        self.assertEqual(len(set(ids)), 1)
        self.assertTrue(tracks[0].confirmed)

    def test_enter_and_leave_events(self):
        for frame in range(3):
            self.tracker.update([self.detection], 1000 + frame)
        scene, events = self.engine.update(self.tracker.active(), 3, now_ms=1003)
        self.assertEqual(events[0]['event_type'], 'object_entered')
        self.assertEqual(scene['primary_track_id'], 1)

        for frame in range(3):
            self.tracker.update([], 2000 + frame)
        _, events = self.engine.update(self.tracker.active(), 6, now_ms=2003)
        self.assertEqual(events[0]['event_type'], 'object_left')

    def test_template_is_bound_to_event(self):
        event = {
            'event_id': 'event-1', 'event_type': 'object_entered',
            'track_id': 7, 'class_name': 'bm', 'severity': 'info',
        }
        result = TemplateGenerator().generate(event, 9)
        self.assertEqual(result['event_id'], 'event-1')
        self.assertEqual(result['scene_revision'], 9)
        self.assertEqual(result['source'], 'template')

    def test_cloud_is_disabled_without_credentials(self):
        generator = CloudGenerator('', '', '')
        self.assertFalse(generator.enabled)
        self.assertIsNone(generator.generate({}, {}))

    def test_cloud_result_is_bound_to_event(self):
        generator = CloudGenerator('https://example.test', 'key', 'model')
        generator._request = lambda event, scene: '检测到比赛目标'
        event = {
            'event_id': 'event-2', 'event_type': 'object_entered',
            'track_id': 3, 'class_name': 'bm', 'severity': 'info',
        }
        result = generator.generate(event, {'revision': 11})
        self.assertEqual(result['event_id'], 'event-2')
        self.assertEqual(result['scene_revision'], 11)
        self.assertEqual(result['source'], 'cloud')

    def test_visual_understanding_disabled_without_key(self):
        client = VisionUnderstandingClient(
            'https://qianfan.baidubce.com/v2', '', 'ernie-4.5-turbo-vl')
        self.assertFalse(client.enabled)
        self.assertIsNone(client.confirm(b'jpeg', {}, ['bm']))

    def test_visual_understanding_rejects_unknown_class(self):
        result = VisionUnderstandingClient.validate_result({
            'confirmed': True, 'class_name': 'unknown',
            'confidence': 1.4, 'reason': 'test'}, ['bm'])
        self.assertFalse(result['confirmed'])
        self.assertIsNone(result['class_name'])
        self.assertEqual(result['confidence'], 1.0)

    def test_tracker_keeps_raw_and_smoothed_confidence(self):
        self.tracker.update([self.detection], 1000)
        lower = Detection(0, 'bm', [10, 10, 110, 110], 0.5)
        track = self.tracker.update([lower], 1001)[0]
        self.assertEqual(track.raw_confidence, 0.5)
        self.assertEqual(track.confidence, 0.74)

    def test_confidence_policy_boundaries(self):
        policy = ConfidencePolicy(0.25, 0.60)
        self.assertEqual(policy.state(0.2499), 'suppressed')
        self.assertEqual(policy.state(0.25), 'review')
        self.assertEqual(policy.state(0.5999), 'review')
        self.assertEqual(policy.state(0.60), 'accepted')

    def test_confidence_policy_fuses_visual_result(self):
        policy = ConfidencePolicy()
        candidate = {'class_name': 'bm', 'track_confidence': 0.5}
        accepted = policy.fuse(candidate, {
            'confirmed': True, 'class_name': 'bm', 'confidence': 0.9})
        self.assertEqual(accepted['decision'], 'accepted')
        self.assertEqual(accepted['effective_confidence'], 0.68)

        rejected = policy.fuse(candidate, {
            'confirmed': False, 'class_name': None, 'confidence': 0.8})
        self.assertEqual(rejected['decision'], 'rejected')
        self.assertEqual(rejected['effective_confidence'], 0.2)

    def test_confidence_policy_validates_thresholds(self):
        with self.assertRaises(ValueError):
            ConfidencePolicy(0.6, 0.6)

    def test_hud_card_falls_back_without_credentials(self):
        generator = HudCardGenerator('', '', '')
        context = {
            'event_id': 'event-3', 'track_id': 4, 'scene_revision': 12,
            'class_name': 'nc', 'display_name': 'nc', 'confidence': 0.68,
        }
        card = generator.generate(context)
        self.assertEqual(card['type'], 'hud_card')
        self.assertEqual(card['source'], 'template')
        self.assertEqual(card['track_id'], 4)
        self.assertEqual(card['class_name'], 'nc')

    def test_hud_card_binds_trusted_context(self):
        generator = HudCardGenerator('https://example.test', 'key', 'ernie-5.1')
        generator._request = lambda context: {
            'title': '赛道目标', 'summary': '检测到目标',
            'facts': [{'label': '状态', 'value': '已确认'}],
        }
        context = {
            'event_id': 'event-4', 'track_id': 8, 'scene_revision': 15,
            'class_name': 'bm', 'display_name': 'bm', 'confidence': 0.72,
        }
        card = generator.generate(context)
        self.assertEqual(card['event_id'], 'event-4')
        self.assertEqual(card['track_id'], 8)
        self.assertEqual(card['scene_revision'], 15)
        self.assertEqual(card['source'], 'ernie-5.1')

    def test_hud_card_rejects_invalid_model_fields(self):
        with self.assertRaises(ValueError):
            HudCardGenerator.validate_content({
                'title': '目标', 'summary': '检测到目标', 'facts': [],
                'track_id': 999,
            })

    def test_hud_card_rejects_oversized_facts(self):
        with self.assertRaises(ValueError):
            HudCardGenerator.validate_content({
                'title': '目标', 'summary': '检测到目标',
                'facts': [{'label': '类别', 'value': '目标'}] * 4,
            })


if __name__ == '__main__':
    unittest.main()
