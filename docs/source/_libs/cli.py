from functools import wraps

import click


def _wrap_func_as_cli(func):
    @wraps(func)
    def _new_func(*args, **kwargs):
        @click.command()
        @click.option('--output', '-o', 'save_as', type=click.Path(dir_okay=False), required=True,
                      help='Output path of image file.', show_default=True)
        def _execute(save_as):
            func(*args, save_as=save_as, **kwargs)

        _execute()

    return _new_func
