#!/usr/bin/env python3

import unittest

from perception_hamer.src.p5_async_runtime import (
    HamerContextState, LatestOnlySlot, P5CapturePacket, SequentialCaptureQueue,
)


class P5RuntimeTest(unittest.TestCase):
    def test_hamer_is_latest_only_and_cannot_publish_orientation(self):
        slot = LatestOnlySlot()
        slot.publish(1); slot.publish(2); slot.publish(3)
        version, value = slot.get_after(0)
        self.assertEqual((version, value), (3, 3))
        self.assertEqual(slot.stats["overwritten"], 2)
        state = HamerContextState()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            state.update({"global_orient": [1, 0, 0]})

    def test_kabsch_queue_reports_overflow_instead_of_silent_backlog(self):
        fifo = SequentialCaptureQueue(capacity=2)
        self.assertTrue(fifo.publish(P5CapturePacket(1, None, None, 1)))
        self.assertTrue(fifo.publish(P5CapturePacket(2, None, None, 2)))
        self.assertFalse(fifo.publish(P5CapturePacket(3, None, None, 3)))
        self.assertEqual(fifo.dropped, 1)


if __name__ == "__main__":
    unittest.main()
