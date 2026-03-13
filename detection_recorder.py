#!/usr/bin/env python3
"""
ROS 2 Detection Recorder
Subscribes to detection topics from any of the three detectors and saves
all data to a timestamped JSON file for later integration (no ROS needed to read it).

Topics recorded per source:
  DETR (config)   /detr/detections  /detr/detection_depths  /detr/annotated_image
  DETR (all)      /detr_all/...
  SegFormer       /segformer/...

Usage:
  python3 detection_recorder.py                        # record from /detr (default)
  python3 detection_recorder.py --source detr_all
  python3 detection_recorder.py --source segformer
  python3 detection_recorder.py --source all           # record all three at once
  python3 detection_recorder.py --source detr --images # also save annotated images
  python3 detection_recorder.py --output my_session    # custom output folder name

Press Ctrl+C to stop — data is saved automatically on exit.
"""

import argparse
import json
import os
import cv2
from datetime import datetime

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge


# ── Topic namespaces for each detector ────────────────────────────────────────
SOURCES = {
    'detr':      '/detr',
    'detr_all':  '/detr_all',
    'segformer': '/segformer',
}


# ──────────────────────────────────────────────────────────────────────────────

class DetectionRecorder(Node):
    """
    Subscribes to detection topics and records every frame to memory,
    then flushes to JSON on shutdown.
    """

    def __init__(self, sources: list, output_dir: str, save_images: bool = False):
        super().__init__('detection_recorder')

        self.bridge      = CvBridge()
        self.save_images = save_images
        self.output_dir  = output_dir
        self.records     = []          # accumulated frame records
        self._depths     = {}          # prefix -> latest depth list (parallel to detections)
        self.frame_count = 0

        os.makedirs(output_dir, exist_ok=True)
        if save_images:
            os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

        for src in sources:
            prefix = SOURCES[src]
            self.get_logger().info(f'Subscribing to {prefix}/*')

            # depths cache (arrives just before or after detections)
            self.create_subscription(
                Float32MultiArray,
                f'{prefix}/detection_depths',
                lambda msg, p=prefix: self._on_depths(msg, p),
                10,
            )

            # annotated image (optional)
            if save_images:
                self.create_subscription(
                    RosImage,
                    f'{prefix}/annotated_image',
                    lambda msg, p=prefix: self._on_image(msg, p),
                    10,
                )

            # detections — main trigger that writes a record
            self.create_subscription(
                Detection2DArray,
                f'{prefix}/detections',
                lambda msg, p=prefix: self._on_detections(msg, p),
                10,
            )

        self.get_logger().info(f'Output folder : {output_dir}/')
        self.get_logger().info('Ctrl+C to stop and save.')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_depths(self, msg: Float32MultiArray, prefix: str):
        self._depths[prefix] = list(msg.data)

    def _on_image(self, msg: RosImage, prefix: str):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ts    = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            name  = prefix.lstrip('/')
            path  = os.path.join(self.output_dir, 'images', f'{name}_{ts}.jpg')
            cv2.imwrite(path, frame)
        except Exception as e:
            self.get_logger().warn(f'Image save error: {e}')

    def _on_detections(self, msg: Detection2DArray, prefix: str):
        depths  = self._depths.get(prefix, [])
        source  = prefix.lstrip('/')

        detections = []
        for i, det in enumerate(msg.detections):
            hyp = det.results[0].hypothesis if det.results else None
            detections.append({
                'class_id':   hyp.class_id if hyp else 'unknown',
                'confidence': round(float(hyp.score), 4) if hyp else 0.0,
                'bbox_center_x': round(float(det.bbox.center.position.x), 2),
                'bbox_center_y': round(float(det.bbox.center.position.y), 2),
                'bbox_width':    round(float(det.bbox.size_x), 2),
                'bbox_height':   round(float(det.bbox.size_y), 2),
                'depth_m':    round(float(depths[i]), 3) if i < len(depths) else None,
            })

        self.records.append({
            'frame':        self.frame_count,
            'ros_stamp_s':  msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'recorded_at':  datetime.now().isoformat(),
            'source':       source,
            'n_detections': len(detections),
            'detections':   detections,
        })
        self.frame_count += 1

        if self.frame_count % 50 == 0:
            total_dets = sum(r['n_detections'] for r in self.records)
            self.get_logger().info(
                f'[{source}] {self.frame_count} frames  |  {total_dets} total detections'
            )

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self):
        if not self.records:
            self.get_logger().warn('No data recorded.')
            return

        total_dets = sum(r['n_detections'] for r in self.records)
        sources    = list({r['source'] for r in self.records})

        payload = {
            'metadata': {
                'recorded_at':      datetime.now().isoformat(),
                'sources':          sources,
                'total_frames':     self.frame_count,
                'total_detections': total_dets,
                'images_saved':     self.save_images,
            },
            'frames': self.records,
        }

        path = os.path.join(self.output_dir, 'detections.json')
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

        self.get_logger().info('─' * 50)
        self.get_logger().info(f'Saved  : {path}')
        self.get_logger().info(f'Frames : {self.frame_count}')
        self.get_logger().info(f'Dets   : {total_dets}')
        self.get_logger().info('─' * 50)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Record ROS 2 detection topics to JSON.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--source', default='detr',
        choices=['detr', 'detr_all', 'segformer', 'all'],
        help='Detector to record from (default: detr)',
    )
    parser.add_argument(
        '--output', default=None,
        help='Output folder name (default: ros_recording_<timestamp>)',
    )
    parser.add_argument(
        '--images', action='store_true',
        help='Also save annotated images as JPEGs',
    )
    args = parser.parse_args()

    sources    = list(SOURCES.keys()) if args.source == 'all' else [args.source]
    output_dir = args.output or f'ros_recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    rclpy.init()
    node = DetectionRecorder(sources, output_dir, save_images=args.images)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
