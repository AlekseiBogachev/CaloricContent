import logging

import click
from omegaconf import OmegaConf

from caloric_content.log import setup_logging
from caloric_content.training import set_random_state

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    "-c",
    default="config.yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the config.yaml containing project configuration.",
)
@click.pass_context
def cli(ctx, config):
    """Run CaloricContent CLI."""
    config_dict = OmegaConf.load(config)
    ctx.obj = config_dict
    set_random_state(seed=config.SEED)
    setup_logging(config_dict, logger_name=__name__)
    logger.info(f"Start CLI. Config loaded from {config}")


if __name__ == "__main__":
    cli()
