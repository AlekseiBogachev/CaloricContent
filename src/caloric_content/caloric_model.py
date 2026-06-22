import logging

import torch
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class CaloricModelConfig(PretrainedConfig):
    model_type = "caloric_model"

    def __init__(
        self,
        image_backbone="google/vit-base-patch16-224",
        text_backbone="google-bert/bert-base-uncased",
        embed_dim=768,
        num_heads=8,
        dropout=0.1,
        train_n_layers=1,
        return_dict=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.train_n_layers = train_n_layers
        self.return_dict = return_dict


class CaloricModel(PreTrainedModel):
    config_class = CaloricModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.image_encoder = AutoModel.from_pretrained(
            config.image_backbone,
            add_pooling_layer=False,
        )
        logger.info(f"Loaded image backbone {config.image_backbone}")

        self.text_encoder = AutoModel.from_pretrained(
            config.text_backbone,
            add_pooling_layer=False,
        )
        logger.info(f"Loaded text backbone {config.text_backbone}")

        image_embed_size = self.image_encoder.config.hidden_size
        text_embed_size = self.text_encoder.config.hidden_size

        if config.embed_dim == image_embed_size:
            self.image_proj = torch.nn.Identity()
        else:
            self.image_proj = torch.nn.Linear(
                image_embed_size, config.embed_dim
            )

        if config.embed_dim == text_embed_size:
            self.text_proj = torch.nn.Identity()
        else:
            self.text_proj = torch.nn.Linear(text_embed_size, config.embed_dim)

        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.LayerNorm(config.embed_dim + 1),
            torch.nn.Linear(config.embed_dim + 1, 512),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(512, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 1),
        )

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        logger.info("Freezed text backbone.")

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        logger.info("Freezed image backbone.")

        if config.train_n_layers > 0:
            for param in self.image_encoder.layers[
                -1 * config.train_n_layers :
            ].parameters():
                param.requires_grad = True
            for param in self.image_encoder.layernorm.parameters():
                param.requires_grad = True
            logger.info(
                f"Unfreezed last {config.train_n_layers} layers "
                "and last layernorm in image backbone."
            )

        self.criterion = torch.nn.L1Loss()

        self.return_dict = config.return_dict

        logger.info("Initialized CaloricModel")

        return None

    def forward(
        self, pixel_values, numeric, input_ids, attention_mask, labels=None
    ):
        imag_embeddings = self.image_encoder(pixel_values=pixel_values)
        text_embeddings = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        imag_embeddings = imag_embeddings.last_hidden_state
        text_embeddings = text_embeddings.last_hidden_state

        imag_embeddings = self.image_proj(imag_embeddings)
        text_embeddings = self.text_proj(text_embeddings)

        key_padding_mask = attention_mask == 0
        fused_embeddings, _ = self.cross_attention(
            query=imag_embeddings,
            key=text_embeddings,
            value=text_embeddings,
            key_padding_mask=key_padding_mask,
        )

        logits = self.regressor(
            torch.cat([fused_embeddings[:, 0, :], numeric], dim=-1)
        )

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        if loss is not None:
            output_dict = {"loss": loss, "logits": logits}
        else:
            output_dict = {"logits": logits}

        if self.return_dict:
            return output_dict
        else:
            return tuple(output_dict.values())
