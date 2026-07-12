import unittest
from collections import namedtuple

from camera_detection.intelligence import (
    CloudGenerator, IoUTracker, SceneEngine, TemplateGenerator)


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


if __name__ == '__main__':
    unittest.main()