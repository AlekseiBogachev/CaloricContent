import logging

import click
from omegaconf import OmegaConf

from caloric_content.log import setup_logging
from caloric_content.training import (
    run_predict,
    run_test,
    run_training,
    set_random_state,
)

logger = logging.getLogger(__name__)


@click.command(name="train")
@click.pass_obj
def train_cmd(config_dict):
    """Run model training."""
    run_training(config_dict)


@click.command(name="test")
@click.pass_obj
def test_cmd(config_dict):
    """Run model evaluation on test set."""
    run_test(config_dict)


@click.command(name="predict")
@click.pass_obj
def predict_cmd(config_dict):
    """Run inference on new data."""
    run_predict(config_dict)


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
    set_random_state(seed=config_dict.get("SEED", 42))
    setup_logging(config_dict, logger_name=__name__)
    logger.info(f"Start CLI. Config loaded from {config}")


cli.add_command(train_cmd)
cli.add_command(test_cmd)
cli.add_command(predict_cmd)


if __name__ == "__main__":
    cli()
