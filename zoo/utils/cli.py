import click
from click.core import Context, Option

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


def print_version(module, ctx: Context, param: Option, value: bool) -> None:
    """
    Print version information of cli
    :param module: current module using this cli.
    :param ctx: click context
    :param param: current parameter's metadata
    :param value: value of current parameter
    """
    _ = param
    if not value or ctx.resilient_parsing:
        return  # pragma: no cover

    click.echo(f'Module utils of {module}')
    ctx.exit()
