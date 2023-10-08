import random

from benchmark import BaseBenchmark, create_plot_cli

from imgutils.detect import detect_text


class TextDetectBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from imgutils.detect.text import _open_text_detect_model
        _ = _open_text_detect_model(self.model)

    def unload(self):
        from imgutils.detect.text import _open_text_detect_model
        _open_text_detect_model.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_text(image_file, model=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            (
                'dbnet_resnet18_fpnc_1200e_icdar2015',
                TextDetectBenchmark('dbnet_resnet18_fpnc_1200e_icdar2015')
            ),
            (
                'dbnet_resnet18_fpnc_1200e_totaltext',
                TextDetectBenchmark('dbnet_resnet18_fpnc_1200e_totaltext')
            ),
            (
                'dbnet_resnet50-oclip_fpnc_1200e_icdar2015',
                TextDetectBenchmark('dbnet_resnet50-oclip_fpnc_1200e_icdar2015')
            ),
            (
                'dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015',
                TextDetectBenchmark('dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015')
            ),
            (
                'dbnetpp_resnet50_fpnc_1200e_icdar2015',
                TextDetectBenchmark('dbnetpp_resnet50_fpnc_1200e_icdar2015')
            ),
        ],
        title='Benchmark for Text Detections',
        run_times=10,
        try_times=20,
    )()
