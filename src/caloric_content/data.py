import logging
from pathlib import Path
import random

import albumentations as A
from datasets import Dataset, DatasetDict, Image
import pandas as pd


logger = logging.getLogger(__name__)


def get_dataset_dict(config):
    logger.debug(f"Start parsing raw data from {config.data_dir}")
    data_dir = Path(config.data_dir)
    ingr_separator = config["text_aug"]["ingr_separator"]

    dish_df = pd.read_csv(data_dir.joinpath("dish.csv"))
    logger.debug("Read dish.csv")

    ingridients_df = pd.read_csv(data_dir.joinpath("ingredients.csv"))
    logger.debug("Read ingredients.csv")

    dish_df = (
        dish_df
        .assign(ingredients=dish_df.ingredients.str.split(";"))
        .explode("ingredients")
    )
    dish_df["ingr_id"] = pd.to_numeric(
        dish_df.ingredients.str.extract("(\d+)$", expand=False)
    )
    dish_df["image"] = (
        dish_df
        .apply(
            lambda row: str(
                data_dir.joinpath("images", row.dish_id, "rgb.png")
            ),
            axis="columns",
        )
    )
    logger.debug("Created dish_df.")

    ingridients_df = (
        pd
        .merge(
            left=dish_df[["dish_id", "ingr_id"]],
            right=ingridients_df,
            left_on="ingr_id",
            right_on="id",
            how="inner",
            validate="many_to_one",
        )
        .drop(columns=["id", "ingr_id"])
        .groupby(by="dish_id")
        .ingr
        .apply(lambda x: ingr_separator.join(x))
        .reset_index()
    )
    ingridients_df["ingr_count"] = ingridients_df.ingr.str.split(ingr_separator).apply(len)
    logger.debug("Created ingridients_df.")

    dish_df = (
        pd.merge(
            left=dish_df.drop(columns=["ingredients", "ingr_id"]).drop_duplicates(),
            right=ingridients_df,
            on="dish_id",
            how="inner",
            validate="one_to_one",
        )
        .drop(columns="dish_id")
        .rename(columns={"ingr": "ingredients"})
        .sort_values(by=["split", "image"])
    )
    logger.debug("Merged dish_df and ingridients_df.")

    dish_dict = {
        split: (
            dish_df
            .query("split == @split")
            .drop(columns="split")
            .to_dict(orient="list")
        )
        for split
        in ["train", "test"]
    }
    logger.debug("Created dataset_dict")

    return dish_dict


def get_dataset(config):
    dataset_dict = get_dataset_dict(config)

    train_dataset = Dataset.from_dict(dataset_dict["train"])
    logger.debug("Created train set.")
    test_dataset = Dataset.from_dict(dataset_dict["test"])
    logger.debug("Created test set.")

    train_dataset = train_dataset.train_test_split(
        test_size=len(test_dataset), seed=config.SEED
    )
    logger.debug("Devided train set to train and valid sets.")

    dataset = DatasetDict({
        "train": train_dataset["train"].cast_column("image", Image(mode="RGB")),
        "valid": train_dataset["test"].cast_column("image", Image(mode="RGB")),
        "test": test_dataset.cast_column("image", Image(mode="RGB")),
    })
    logger.info(
        "Created Hugging Face dataset: "
        f"train length = {len(dataset['train'])}, "
        f"valid length = {len(dataset['valid'])}, "
        f"test length = {len(dataset['test'])}."
    )

    return dataset


def get_images_transforms(config, split="train"):
    transforms = list()
    logger.debug("Start building images transfroms from config.")

    albums = config.get("albumentations", list())
    if albums:
        albums = albums.get(split, list())
    for album_params in albums:
        album_cls = getattr(A, album_params.pop("name"))
        transforms.append(album_cls(**album_params))
        logger.debug(f"Add transform: {transforms[-1]}")

    transforms = A.Compose(transforms, seed=config.SEED)
    logger.info(f"Created transforms for images:\n{transforms}")

    return transforms


def get_total_mass_transform(config):
    mean = config["total_mass"]["mean"]
    std = config["total_mass"]["std"]

    def standardize(value):
        return (value - mean) / std

    return standardize


def get_text_transforms(config):
    ingr_sep = config["text_aug"]["ingr_separator"]

    drop_prob = config["text_aug"]["drop_ingr"]["p"]
    drop_frac = config["text_aug"]["drop_ingr"]["f_ingr"]

    shuffle_prob = config["text_aug"]["shuffle_ingr"]["p"]

    def transforms(text):
        items = text.split(ingr_sep)

        if random.random() < drop_prob:
            random.shuffle(items)
            n_items = len(items) - int(drop_frac * len(items))
            items = items[:n_items]
        elif random.random() < shuffle_prob:
            random.shuffle(items)
        else:
            return text

        return ingr_sep.join(items)

    return transforms



