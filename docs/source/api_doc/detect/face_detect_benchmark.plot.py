import random

from benchmark import BaseBenchmark, create_plot_cli
from imgutils.detect import detect_faces


class FaceDetectBenchmark(BaseBenchmark):
    def __init__(self, level, version):
        BaseBenchmark.__init__(self)
        self.level = level
        self.version = version

    def load(self):
        from imgutils.detect.face import _open_face_detect_model
        _ = _open_face_detect_model(level=self.level, version=self.version)

    def unload(self):
        from imgutils.detect.face import _open_face_detect_model
        _open_face_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_faces(image_file, level=self.level, version=self.version)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('face v1.3 (yolov8s)', FaceDetectBenchmark('s', 'v1.3')),
            ('face v1.3 (yolov8n)', FaceDetectBenchmark('n', 'v1.3')),
            ('face v1 (yolov8s)', FaceDetectBenchmark('s', 'v1')),
            ('face v1 (yolov8n)', FaceDetectBenchmark('n', 'v1')),
            ('face v0 (yolov8s)', FaceDetectBenchmark('s', 'v0')),
            ('face v0 (yolov8n)', FaceDetectBenchmark('n', 'v0')),
        ],
        title='Benchmark for Anime Face Detections',
        run_times=10,
        try_times=20,
    )()
