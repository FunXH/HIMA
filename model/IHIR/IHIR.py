import torch
import torch.nn.functional as F

from .hash.hash import HashLayer as InstanceHashProjector
from attacked.models.base import BaseModel
from attacked.common.register import registry
from .loss.triplet_loss import TripletLoss
from .distance import SetwiseDistance

@registry.register_model("IHIR")
class IHIR(BaseModel):

    DEFAULT_CONFIG_FILE = {"base": "attacked/configs/IHIR/base.yaml"}

    def __init__(
                self,
                config,
                code_width=16,
                backbone_path="./ViT-B-32.pt",
                ranking_margin=0.1,
                training_sample_count=10000,
                context_token_count=32,
                latent_instance_count=8,
                dropout_rate=0.3,
                projection_mode: str="softmax",
                pooling_strategy: str="mean",
                global_token_weight=0.7,
                **unused_options):
        super().__init__()
        self.config = config
        model_config = config.model if hasattr(config, "model") else config
        self.projection_mode = projection_mode
        state_dict, self.feature_backbone = self.load_clip(clipPath=backbone_path, return_patches=True)
        encoder_width = state_dict["text_projection"].shape[1]
        image_token_count = state_dict.get("visual.positional_embedding", torch.empty(50)).shape[0] - 1
        self.image_token_count = image_token_count
        self.context_token_count = context_token_count
        self.global_token_weight = global_token_weight
        self.instance_hash_projector = InstanceHashProjector(
            visual_tokens=image_token_count,
            txt_tokens=context_token_count,
            feature_size=encoder_width,
            outputDim=code_width,
            setDim=latent_instance_count,
            dropout=dropout_rate,
            hash_func_=projection_mode,
            merge_func=pooling_strategy,
        )
        self.output_dim = code_width
        self.ranking_criterion = TripletLoss(reduction="mean")
        self.setwise_distance = SetwiseDistance(img_set_size=latent_instance_count, txt_set_size=latent_instance_count, 
                                        denominator=2.0, temperature=16.0,
                                        temperature_txt_scale=1.0, mode=model_config.distance.get("mode", "chamfer"))
        self.instance_geometry_weights = {
            "set_size": latent_instance_count,
            "margin": model_config.chamfer.get("margin", 0.5),
            "semi_hard_triplet": model_config.chamfer.get("semi_hard_triplet", False),
            "max_violation": model_config.chamfer.get("max_violation", True),
            "mmd_alpha": model_config.chamfer.get("mmd_alpha", 0.01),
            "unif_alpha": model_config.chamfer.get("unif_alpha", 0.01),
            "mmd_gamma": model_config.chamfer.get("mmd_gamma", 0.5),
            "token_triplet_margin": model_config.chamfer.get("token_triplet_margin", 0.2)
        }
        self.code_objective_weights = {
            "triplet_alpha": model_config.hash_pars.get("triplet_alpha", 1),
            "infonce_alpha": model_config.hash_pars.get("infonce_alpha", 0),
            "quan_alpha": model_config.hash_pars.get("quan_alpha", 0.001),
            "hash_triplet_alpha": model_config.hash_pars.get("hash_triplet_alpha", 0.5),
        }
        self.ranking_margin = model_config.hash_pars.get("triplet_margin", ranking_margin)

    def extract_image_instances(self, image):

        cls_embedding, patch_embeddings, attention_map = self.feature_backbone.encode_image(image)
        if patch_embeddings.shape[0] != self.image_token_count and patch_embeddings.shape[1] != self.image_token_count:
            instance_tokens = torch.vstack([cls_embedding.unsqueeze(dim=0), patch_embeddings])
        else:
            instance_tokens = patch_embeddings
        instance_tokens = (
            instance_tokens.permute(1, 0, 2)
            if instance_tokens.shape[1] != self.image_token_count
            else instance_tokens
        )

        instance_embeddings, instance_codes = self.instance_hash_projector.encode_img(instance_tokens)
        return instance_embeddings, instance_codes

    def extract_text_instances(self, text):
        _, token_sequence, _, _ = self.feature_backbone.encode_text(text)
        token_sequence = (
            token_sequence.permute(1, 0, 2)
            if token_sequence.shape[1] != self.context_token_count
            else token_sequence
        )
        instance_embeddings, instance_codes = self.instance_hash_projector.encode_txt(token_sequence)
        return instance_embeddings, instance_codes

    def encode_image(self, image):
        return self.extract_image_instances(image)

    def encode_text(self, text):
        return self.extract_text_instances(text)
    
    @classmethod
    def from_config(cls, cfg, output_dim=64, training_sample_count=10000, text_token_count=32):
        model_config = cfg.model if hasattr(cfg, "model") else cfg
        backbone_path = model_config.get("clip_path", "./ViT-B-32.pt")
        ranking_margin = model_config.get("triplet_margin", 0.1)
        latent_instance_count = model_config.get("setDim", 64)
        dropout_rate = model_config.get("dropout", 0.3)
        projection_mode = model_config.get("hash_func", "softmax")
        pooling_strategy = model_config.get("merge_func", "mean")
        global_token_weight = model_config.get("cls_alpha", "0.7")
        if "softmax" in projection_mode:
            output_dim *= 2 

        ihir_probe = cls(
            config=cfg,
            code_width=output_dim,
            backbone_path=backbone_path,
            ranking_margin=ranking_margin,
            training_sample_count=training_sample_count,
            context_token_count=text_token_count,
            latent_instance_count=latent_instance_count,
            dropout_rate=dropout_rate,
            projection_mode=projection_mode,
            pooling_strategy=pooling_strategy,
            global_token_weight=global_token_weight
        )
        return ihir_probe
    
    def forward(self, image, text, labels=None, indexs=None, return_loss=False):
        image_instances, image_codes = self.extract_image_instances(image)
        text_instances, text_codes = self.extract_text_instances(text)
        
        if return_loss:
            return self.object_function(
                img_hash=image_codes,
                txt_hash=text_codes,
                labels=labels,
                indexs=indexs,
                img_embeds=image_instances,
                txt_embeds=text_instances,
            )
        
        return image_instances, image_codes, text_instances, text_codes
    
    def categorical_code_regularizer(self, code_tensor):
        if len(code_tensor.shape) < 3:
            code_tensor = code_tensor.view(code_tensor.shape[0], -1, 2)
        
        saturation_penalty = 1 - torch.pow(2 * code_tensor - 1, 2).mean()
        return saturation_penalty

    def sign_consistency_regularizer(self, code_tensor):
        
        detached_sign = torch.sign(code_tensor.detach())
        return F.mse_loss(code_tensor, detached_sign)

    def pairwise_code_ranking_terms(self, image_codes, text_codes, label_matrix, indexs=None, **unused_options):
        assert self.ranking_criterion is not None, "please initialing self.ranking_criterion before computing."
        image_to_text_loss = self.ranking_criterion(
            image_codes, label_matrix, target=text_codes, margin=self.ranking_margin
        )
        text_to_image_loss = self.ranking_criterion(
            text_codes, label_matrix, target=image_codes, margin=self.ranking_margin
        )

        if self.projection_mode == "softmax":
            image_quantization_loss = self.categorical_code_regularizer(image_codes)
            text_quantization_loss = self.categorical_code_regularizer(text_codes)
        elif self.projection_mode == "tanh":
            image_quantization_loss = self.sign_consistency_regularizer(image_codes)
            text_quantization_loss = self.sign_consistency_regularizer(text_codes)

        return image_to_text_loss, text_to_image_loss, image_quantization_loss, text_quantization_loss
    
    
    def instance_alignment_loss(
            self,
            image_instance_embeddings,
            image_instance_codes,
            text_instance_embeddings,
            text_instance_codes,
            labels=None,
            indexs=None,
            **unused_options):
        
        assert self.setwise_distance is not None, "setwise distance must initialize before computing."
        
        def compact_rbf_kernel(left_embeddings, right_embeddings, gamma):
            """RBF kernel that does not cause memory shortage"""
            pairwise_distance = torch.cdist(left_embeddings, right_embeddings)
            return torch.exp(-gamma * pairwise_distance)

        # for Maximum Mean Discrepancy
        def distribution_alignment_loss(source_embeddings, target_embeddings, gamma=None, reduction='mean'):
            if gamma is None:
                gamma = 1./source_embeddings.size(-1)
            if reduction=='mean':
                loss = (
                    compact_rbf_kernel(source_embeddings, source_embeddings, gamma).mean()
                    - 2 * compact_rbf_kernel(source_embeddings, target_embeddings, gamma).mean()
                    + compact_rbf_kernel(target_embeddings, target_embeddings, gamma).mean()
                )
            else:
                loss = (
                    compact_rbf_kernel(source_embeddings, source_embeddings, gamma).sum()
                    - 2 * compact_rbf_kernel(source_embeddings, target_embeddings, gamma).sum()
                    + compact_rbf_kernel(target_embeddings, target_embeddings, gamma).sum()
                )
            return loss

        # for token embeds diverse
        def instance_dispersion_loss(instance_embeddings, instance_count, scale=20):
            if instance_count == 1:
                return 0.0
            rbf_values = torch.exp(-scale * torch.cdist(instance_embeddings, instance_embeddings).pow(2))
            upper_triangle_mask = torch.triu(
                torch.ones(rbf_values.shape[1], rbf_values.shape[1]), diagonal=1
            )
            upper_triangle_mask = upper_triangle_mask.unsqueeze(0).expand(
                rbf_values.shape[0], -1, -1
            ).to(instance_embeddings.device)
            rbf_values = torch.where(upper_triangle_mask == 1, rbf_values, torch.zeros_like(rbf_values))
            loss = torch.stack([row.sum() for row in rbf_values]) / (instance_count * (instance_count - 1) * 0.5)
            return loss.mean()

        image_instance_embeddings = image_instance_embeddings.view(-1, image_instance_embeddings.shape[-1])
        text_instance_embeddings = text_instance_embeddings.view(-1, text_instance_embeddings.shape[-1])
        image_instance_embeddings = F.normalize(image_instance_embeddings, dim=-1)
        text_instance_embeddings = F.normalize(text_instance_embeddings, dim=-1)
        total_loss = 0

        image_text_similarity = self.setwise_distance.compute(
            img_embs=image_instance_embeddings, txt_embs=text_instance_embeddings
        )
        text_image_similarity = self.setwise_distance.compute(
            img_embs=text_instance_embeddings, txt_embs=image_instance_embeddings
        )
        image_text_instance_loss = self.ranking_criterion(
            None,
            labels,
            target=None,
            distance=torch.clamp(1 - image_text_similarity, 0),
            margin=self.instance_geometry_weights["token_triplet_margin"],
        )
        text_image_instance_loss = self.ranking_criterion(
            None,
            labels,
            target=None,
            distance=torch.clamp(1 - text_image_similarity, 0),
            margin=self.instance_geometry_weights["token_triplet_margin"],
        )
        
        mmd_loss = distribution_alignment_loss(
            image_instance_embeddings,
            text_instance_embeddings,
            gamma=self.instance_geometry_weights["mmd_gamma"],
        )
        dispersion_loss = instance_dispersion_loss(
            image_instance_embeddings,
            instance_count=self.instance_geometry_weights["set_size"],
        ) + instance_dispersion_loss(
            text_instance_embeddings,
            instance_count=self.instance_geometry_weights["set_size"],
        )

        total_loss += (
            (image_text_instance_loss + text_image_instance_loss) / 2
            * self.code_objective_weights["triplet_alpha"]
            + self.instance_geometry_weights["mmd_alpha"] * mmd_loss
            + self.instance_geometry_weights["unif_alpha"] * dispersion_loss
        )

        
        code_image_text_loss, code_text_image_loss, image_quantization_loss, text_quantization_loss = (
            self.pairwise_code_ranking_terms(
                image_instance_codes, text_instance_codes, labels, indexs, **unused_options
            )
        )
        code_ranking_loss = (code_image_text_loss + code_text_image_loss) / 2
        quantization_loss = (image_quantization_loss + text_quantization_loss) / 2

        total_loss += (
            code_ranking_loss * self.code_objective_weights["hash_triplet_alpha"]
            + quantization_loss * self.code_objective_weights["quan_alpha"]
        )
        loss_dict = {
            "All loss": total_loss.data,
            "Tokens": {
                "Similarity": {
                    "i2t": image_text_instance_loss.data,
                    "t2i": text_image_instance_loss.data
                    },
                "Maximum Mean Discrepancy": mmd_loss.data,
                # "Diversity": div_loss.data if isinstance(div_loss, torch.Tensor) else div_loss
                },
            "Hash": {
                "Triplet": {
                    "i2t": code_image_text_loss.data if code_image_text_loss is not None else None,
                    "t2i": code_text_image_loss.data if code_text_image_loss is not None else None
                    },
                "Quantization": {
                    "image": image_quantization_loss.data,
                    "text": text_quantization_loss.data
                    }
                }
            }
        
        return total_loss, loss_dict


    def object_function(self, img_embeds, img_hash, txt_embeds, txt_hash, labels=None, indexs=None, **kwags):
        if labels is None:
            labels = torch.ones([img_embeds.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.instance_alignment_loss(
            image_instance_embeddings=img_embeds,
            image_instance_codes=img_hash,
            text_instance_embeddings=txt_embeds,
            text_instance_codes=txt_hash,
            labels=labels,
            indexs=indexs,
            **kwags
        )
