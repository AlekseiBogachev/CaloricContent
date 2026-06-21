import logging
import random
from copy import deepcopy
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Image

logger = logging.getLogger(__name__)


def get_dataset_dict(config):
    logger.debug(f"Start parsing raw data from {config.data_dir}")
    data_dir = Path(config.data_dir)
    ingr_separator = config["text_aug"]["ingr_separator"]

    dish_df = pd.read_csv(data_dir.joinpath("dish.csv"))
    logger.debug("Read dish.csv")

    ingredients_df = pd.read_csv(data_dir.joinpath("ingredients.csv"))
    logger.debug("Read ingredients.csv")

    dish_df = dish_df.assign(
        ingredients=dish_df.ingredients.str.split(";")
    ).explode("ingredients")
    dish_df["ingr_id"] = pd.to_numeric(
        dish_df.ingredients.str.extract("(\d+)$", expand=False)
    )
    dish_df["image"] = dish_df.apply(
        lambda row: str(data_dir.joinpath("images", row.dish_id, "rgb.png")),
        axis="columns",
    )
    logger.debug("Created dish_df.")

    ingredients_df = (
        pd.merge(
            left=dish_df[["dish_id", "ingr_id"]],
            right=ingredients_df,
            left_on="ingr_id",
            right_on="id",
            how="inner",
            validate="many_to_one",
        )
        .drop(columns=["id", "ingr_id"])
        .groupby(by="dish_id")
        .ingr.apply(lambda x: ingr_separator.join(x))
        .reset_index()
    )
    ingredients_df["ingr_count"] = ingredients_df.ingr.str.split(
        ingr_separator
    ).apply(len)
    logger.debug("Created ingredients_df.")

    dish_df = (
        pd.merge(
            left=dish_df.drop(
                columns=["ingredients", "ingr_id"]
            ).drop_duplicates(),
            right=ingredients_df,
            on="dish_id",
            how="inner",
            validate="one_to_one",
        )
        .drop(columns="dish_id")
        .rename(columns={"ingr": "ingredients"})
        .sort_values(by=["split", "image"])
    )
    logger.debug("Merged dish_df and ingredients_df.")

    dish_dict = {
        split: (
            dish_df.query("split == @split")
            .drop(columns="split")
            .to_dict(orient="list")
        )
        for split in ["train", "test"]
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

    dataset = DatasetDict(
        {
            "train": train_dataset["train"].cast_column(
                "image", Image(mode="RGB")
            ),
            "valid": train_dataset["test"].cast_column(
                "image", Image(mode="RGB")
            ),
            "test": test_dataset.cast_column("image", Image(mode="RGB")),
        }
    )
    logger.info(
        "Created Hugging Face dataset: "
        f"train length = {len(dataset['train'])}, "
        f"valid length = {len(dataset['valid'])}, "
        f"test length = {len(dataset['test'])}."
    )

    return dataset


def get_images_transforms(config, split="train"):
    config_dict = deepcopy(config)
    transforms = list()
    logger.debug("Start building images transfroms from config.")

    albums = config_dict.get("albumentations", list())
    if albums:
        albums = albums.get(split, list())
    for album_params in albums:
        album_cls = getattr(A, album_params.pop("name"))
        transforms.append(album_cls(**album_params))
        logger.debug(f"Add transform: {transforms[-1]}")

    transforms = A.Compose(transforms, seed=config_dict.SEED)
    logger.info(f"Created transforms for images:\n{transforms}")

    return transforms


def get_total_mass_transform(config, split="train"):
    mean = config["total_mass"]["mean"]
    std = config["total_mass"]["std"]
    noise_std = config["total_mass"]["noise_std"]

    def standardize(value):
        if noise_std > 0 and split == "train":
            noise = np.random.normal(scale=noise_std)
        else:
            noise = 0
        return (value - mean) / std + noise

    logger.info(
        "Create standartization for 'total_mass' with "
        f"mean={mean} and std={std}"
    )

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

    logger.info(
        f"Create text augmentations with parameters={config['text_aug']}"
    )

    return transforms


def apply_transforms_to_dataset(
    dataset,
    config,
    split="train",
):
    img_transforms = get_images_transforms(config, split)
    if split == "train":
        text_transforms = get_text_transforms(config)
    total_mass_transform = get_total_mass_transform(config, split=split)

    def transform_fn(examples):
        transformed = dict(labels=examples["labels"])
        transformed["pixel_values"] = [
            img_transforms(image=np.array(image.convert("RGB")))["image"]
            for image in examples["image"]
        ]
        if split == "train":
            transformed["text"] = [
                text_transforms(text) for text in examples["ingredients"]
            ]
        else:
            transformed["text"] = examples["ingredients"]

        transformed["numeric"] = [
            total_mass_transform(mass) for mass in examples["total_mass"]
        ]

        return transformed

    dataset = dataset.rename_column("total_calories", "labels")
    dataset.set_transform(transform_fn)

    logger.info(f"Applied transforms for dataset. split={split}")

    return dataset


def collate_fn(batch, tokenizer):
    labels = torch.tensor(
        [item["labels"] for item in batch], dtype=torch.float32
    ).unsqueeze(1)
    images = torch.stack([item["pixel_values"] for item in batch])
    texts = [item["text"] for item in batch]
    numeric_vals = torch.tensor(
        [item["numeric"] for item in batch], dtype=torch.float32
    ).unsqueeze(1)

    tokenized_text = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    )

    return {
        "labels": labels,
        "pixel_values": images,
        "numeric": numeric_vals,
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
    }
