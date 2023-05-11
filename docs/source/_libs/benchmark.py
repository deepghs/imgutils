import glob
import multiprocessing
import os
import time
import warnings
from multiprocessing import Process
from typing import Tuple, List

import click
import matplotlib.pyplot as plt
import numpy as np
import psutil
from hbutils.scale import size_to_bytes_str
from hbutils.string import ordinalize, plural_word
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm

from conf import PROJ_DIR
from plot import INCHES_TO_PIXELS

_DEFAULT_IMAGE_POOL = glob.glob(os.path.join(PROJ_DIR, 'test', 'testfile', 'dataset', '**', '*.jpg'), recursive=True)


class BaseBenchmark:
    def __init__(self):
        self.all_images = _DEFAULT_IMAGE_POOL

    def prepare(self):
        pass

    def load(self):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def run_benchmark(self, run_times):
        logs = []
        current_process = psutil.Process()

        def _record(name):
            logs.append((name, current_process.memory_info().rss, time.time()))

        # make sure the model is downloaded
        self.prepare()
        self.load()
        self.unload()

        _record('<init>')

        self.load()
        _record('<load>')

        for i in tqdm(range(run_times)):
            self.run()
            _record(f'#{i + 1}')

        self.unload()
        _record('<unload>')

        mems = np.array([mem for _, mem, _ in logs])
        mems -= mems[0]
        times = np.array([time_ for _, _, time_ in logs])
        times -= times[0]
        times[1:] = times[1:] - times[:-1]
        labels = np.array([name for name, _, _ in logs])

        return mems, times, labels

    def _run_in_subprocess_share(self, run_times, ret):
        ret['retval'] = self.run_benchmark(run_times)

    def run_in_subprocess(self, run_times: int = 10, try_times: int = 10):
        manager = multiprocessing.Manager()
        full_deltas, full_times, final_labels = [], [], None
        for i in tqdm(range(try_times)):
            ret = manager.dict()
            p = Process(target=self._run_in_subprocess_share, args=(run_times, ret,))
            p.start()
            p.join()
            if p.exitcode != 0:
                raise ChildProcessError(f'Exitcode {p.exitcode} in {self!r}\'s {ordinalize(i + 1)} try.')

            mems, times, labels = ret['retval']
            deltas = mems[1:] - mems[:-1]
            full_deltas.append(deltas)
            full_times.append(times)
            if final_labels is None:
                final_labels = labels

        deltas = np.stack(full_deltas).mean(axis=0)
        final_mems = np.cumsum([0, *deltas])
        final_times = np.stack(full_times).mean(axis=0)

        return final_mems, final_times, final_labels


def create_plot_cli(items: List[Tuple[str, BaseBenchmark]],
                    title: str = 'Unnamed Benchmark Plot', run_times=15, try_times=10,
                    mem_ylog: bool = False, time_ylog: bool = False,
                    figsize=(1080, 600), dpi: int = 300):
    def fmt_size(x, pos):
        _ = pos
        warnings.filterwarnings('ignore')
        return size_to_bytes_str(x, precision=1)

    def fmt_time(x, pos):
        _ = pos
        if x < 1e-6:
            return f'{x * 1e9:.1f}ns'
        elif x < 1e-3:
            return f'{x * 1e6:.1f}μs'
        elif x < 1:
            return f'{x * 1e3:.1f}ms'
        else:
            return f'{x * 1.0:.1f}s'

    @click.command()
    @click.option('--output', '-o', 'save_as', type=click.Path(dir_okay=False), required=True,
                  help='Output path of image file.', show_default=True)
    def _execute(save_as):
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] / INCHES_TO_PIXELS, figsize[1] / INCHES_TO_PIXELS))

        if mem_ylog:
            axes[0].set_yscale('log')
        axes[0].yaxis.set_major_formatter(FuncFormatter(fmt_size))
        axes[0].set_title('Memory Benchmark')
        axes[0].set_ylabel('Memory Usage')

        if time_ylog:
            axes[1].set_yscale('log')
        axes[1].yaxis.set_major_formatter(FuncFormatter(fmt_time))
        axes[1].set_title('Performance Benchmark (CPU)')
        axes[1].set_ylabel('Time Cost')

        labeled = False

        for name, bm in tqdm(items):
            mems, times, labels = bm.run_in_subprocess(run_times, try_times)
            axes[0].plot(mems, label=name)
            axes[1].plot(times, label=name)
            if not labeled:
                axes[0].set_xticks(range(len(labels)), labels, rotation='vertical')
                axes[1].set_xticks(range(len(labels)), labels, rotation='vertical')
                labeled = True

        axes[0].legend()
        axes[0].grid()
        axes[1].legend()
        axes[1].grid()

        fig.suptitle(f'{title}\n'
                     f'(Mean of {plural_word(try_times, "try")}, '
                     f'run for {plural_word(run_times, "time")})')

        fig.tight_layout()
        plt.savefig(save_as, bbox_inches='tight', dpi=dpi, transparent=True)

    return _execute
