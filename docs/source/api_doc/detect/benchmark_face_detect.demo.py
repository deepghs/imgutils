import random

from benchmark import BaseBenchmark, create_plot
from imgutils.detect import detect_faces


class FaceDetectBenchmark(BaseBenchmark):
    def __init__(self, level):
        BaseBenchmark.__init__(self)
        self.level = level

    def load(self):
        from imgutils.detect.face import _open_face_detect_model
        _ = _open_face_detect_model(level=self.level)

    def unload(self):
        from imgutils.detect.face import _open_face_detect_model
        _open_face_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_faces(image_file, level=self.level)


if __name__ == '__main__':
    create_plot(
        [
            ('face (yolov8s)', FaceDetectBenchmark('s')),
            ('face (yolov8n)', FaceDetectBenchmark('n')),
        ],
        save_as='benchmark_face_detect.bm.svg',
        title='Benchmark for Anime Face Detections',
        run_times=10,
        try_times=5,
        figsize=(1080, 600)
    )
