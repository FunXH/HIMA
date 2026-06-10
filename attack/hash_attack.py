import os
from typing import List, Union

import scipy.io as scio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataloader import dataloader
from model.IHIR import IHIR
from model.optimization import BertAdam
from attack.base import TrainBase
from utils import get_args
from utils.calc_utils import calc_map_k_matrix as calc_map_k, calc_soft_map_k


class Trainer(TrainBase):
    def __init__(self, rank=0):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("ViT+GPT!")

        if self.args.model == "DCMHT":
            from attacked.models.DCMHT import DCMHT

            cfg = read_config("attacked/configs/DCMHT/config.yaml")
            self.model = DCMHT(cfg, outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)
            ihir_cfg = read_config("attacked/configs/IHIR/config.yaml")
            self.model2 = IHIR.from_config(
                ihir_cfg, output_dim=self.args.output_dim, text_token_count=self.args.max_words
            ).to(self.rank)

        if self.args.model == "DNPH":
            from attacked.models.DNPH import DNPH

            cfg = read_config("attacked/configs/DNPH/config.yaml")
            self.model = DNPH(cfg, outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)
            ihir_cfg = read_config("attacked/configs/IHIR/config.yaml")
            self.model2 = IHIR.from_config(
                ihir_cfg, output_dim=self.args.output_dim, text_token_count=self.args.max_words
            ).to(self.rank)

        if self.args.model == "DSPH":
            from attacked.models.DSPH import DSPH

            cfg = read_config("attacked/configs/DSPH/config.yaml")
            self.model = DSPH(cfg, outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)
            ihir_cfg = read_config("attacked/configs/IHIR/config.yaml")
            self.model2 = IHIR.from_config(
                ihir_cfg, output_dim=self.args.output_dim, text_token_count=self.args.max_words
            ).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
            pretrained2 = getattr(self.args, "pretrained2", "")
            if pretrained2 and os.path.exists(pretrained2):
                self.model2.load_state_dict(torch.load(pretrained2, map_location=f"cuda:{self.rank}"))

        self.model.float()
        self.model2.float()
        self.model2.eval()
        for param in self.model2.parameters():
            param.requires_grad_(False)

        self.optimizer = BertAdam(
            [
                {"params": self.model.clip.parameters(), "lr": self.args.clip_lr},
                {"params": self.model.hash.parameters(), "lr": self.args.lr},
            ],
            lr=self.args.lr,
            warmup=self.args.warmup_proportion,
            schedule="warmup_cosine",
            b1=0.9,
            b2=0.98,
            e=1e-6,
            t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay,
            max_grad_norm=1.0,
        )

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data, target_data = dataloader(
            captionFile=self.args.caption_file,
            indexFile=self.args.index_file,
            labelFile=self.args.label_file,
            maxWords=self.args.max_words,
            imageResolution=self.args.resolution,
            query_num=self.args.query_num,
            train_num=self.args.train_num,
            seed=self.args.seed,
        )
        self.train_labels = train_data.get_all_label()
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.target_labels = target_data.get_all_label()

        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        self.target_loader = DataLoader(
            dataset=target_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def _hima_arg(self, name: str, default, legacy_name: str = None):
        if hasattr(self.args, name):
            return getattr(self.args, name)
        if legacy_name is not None and hasattr(self.args, legacy_name):
            return getattr(self.args, legacy_name)
        return default

    def _soft_binary_code(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(h)

    def _match_hash_dim(self, a: torch.Tensor, b: torch.Tensor):
        dim = min(a.shape[-1], b.shape[-1])
        return a[..., :dim], b[..., :dim]

    def _normalized_hamming_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a, b = self._match_hash_dim(a, b)
        if b.dim() == 3 and a.dim() == 2:
            a = a.unsqueeze(1)
        return 0.5 * (1.0 - (a * b).mean(dim=-1))

    def _select_reliable_instance_anchors(
        self,
        instance_hashes: torch.Tensor,
        holistic_target_hash: torch.Tensor,
        top_k: int = None,
    ) -> torch.Tensor:
        """Preview placeholder for reliability-aware instance screening.

        The full scoring and anchor selection strategy will be released after
        paper acceptance. This version preserves the public interface and keeps
        the attack pipeline executable with a deterministic top-k placeholder.
        """
        if instance_hashes.dim() == 2:
            instance_hashes = instance_hashes.unsqueeze(1)

        top_k = top_k or self._hima_arg("ihir_topk", self._hima_arg("token", 4), legacy_name="hima_topk")
        top_k = max(1, min(int(top_k), instance_hashes.shape[1]))

        return instance_hashes[:, :top_k, :].detach()

    def _multi_scale_hamming_alignment_loss(
        self,
        adv_hash: torch.Tensor,
        holistic_target_hash: torch.Tensor,
        reliable_anchors: torch.Tensor,
    ):
        """Preview placeholder for the holistic-to-instance Hamming objective.

        The released paper version will include the complete multi-scale
        instance-anchor term. The public preview keeps holistic target alignment
        only, while preserving the same return signature.
        """
        soft_adv = self._soft_binary_code(adv_hash)
        soft_target = self._soft_binary_code(holistic_target_hash.detach())

        loss_holistic = self._normalized_hamming_distance(soft_adv, soft_target).mean()
        loss_instance = loss_holistic.new_zeros(())
        loss_total = loss_holistic
        return loss_total, loss_holistic.detach(), loss_instance.detach()

    def _instance_set_anchor_loss(
        self,
        adv_instance_hashes: torch.Tensor,
        reliable_anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Preview placeholder for text instance-anchor alignment."""
        return adv_instance_hashes.new_zeros(())

    def _text_holistic_instance_loss(
        self,
        adv_text_hash: torch.Tensor,
        adv_text_instance_hashes: torch.Tensor,
        holistic_target_hash: torch.Tensor,
        reliable_anchors: torch.Tensor,
    ):
        loss_total, loss_holistic, loss_anchor = self._multi_scale_hamming_alignment_loss(
            adv_hash=adv_text_hash,
            holistic_target_hash=holistic_target_hash,
            reliable_anchors=reliable_anchors,
        )
        text_instance_weight = float(
            self._hima_arg("ihir_text_instance_lambda", 0.5, legacy_name="hima_text_instance_lambda")
        )
        loss_text_instance = self._instance_set_anchor_loss(
            adv_instance_hashes=adv_text_instance_hashes,
            reliable_anchors=reliable_anchors,
        )
        loss_total = loss_total + text_instance_weight * loss_text_instance
        return loss_total, loss_holistic, loss_anchor, loss_text_instance.detach()

    def _target_text_instance_anchors(
        self,
        image: torch.Tensor,
        target_text: torch.Tensor,
        target_labels: torch.Tensor,
        index: torch.Tensor,
        holistic_target_hash: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            image2 = image.to(dtype=next(self.model2.parameters()).dtype)
            _, _, target_text_instances, _ = self.model2(image2, target_text, target_labels, index, False)
        return self._select_reliable_instance_anchors(target_text_instances, holistic_target_hash)

    def _target_image_instance_anchors(
        self,
        target_image: torch.Tensor,
        text: torch.Tensor,
        target_labels: torch.Tensor,
        index: torch.Tensor,
        holistic_target_hash: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_image2 = target_image.to(dtype=next(self.model2.parameters()).dtype)
            target_image_instances, _, _, _ = self.model2(target_image2, text, target_labels, index, False)
        return self._select_reliable_instance_anchors(target_image_instances, holistic_target_hash)

    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt = super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = super().get_code(self.retrieval_loader, self.args.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, "
            f"MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, MAX MAP(i->t): {self.max_mapi2t}, "
            f"MAX MAP(t->i): {self.max_mapt2i}"
        )

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        result_dict = {
            "q_img": query_img.cpu().detach().numpy(),
            "q_txt": query_txt.cpu().detach().numpy(),
            "r_img": retrieval_img.cpu().detach().numpy(),
            "r_txt": retrieval_txt.cpu().detach().numpy(),
            "q_l": self.query_labels.numpy(),
            "r_l": self.retrieval_labels.numpy(),
        }
        scio.savemat(
            os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"),
            result_dict,
        )
        self.logger.info(f">>>>>> save best {mode_name} data!")

    def image_attack(
        self,
        num,
        image: torch.Tensor,
        target_text: torch.Tensor,
        query_labels,
        index,
        target_features,
        origin_features,
        target_hash,
        origin_hash,
        epsilon,
        alpha,
        num_iter,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> torch.Tensor:
        with torch.no_grad():
            self.model.clip.eval()
            self.model.eval()
            self.model2.eval()
        orig_image = image.clone().detach().to(self.rank)
        adv_image = image.clone().detach().to(self.rank)
        orig_image.requires_grad_(False)

        target_hash = target_hash.detach()
        reliable_anchors = self._target_text_instance_anchors(
            image=orig_image,
            target_text=target_text,
            target_labels=query_labels,
            index=index,
            holistic_target_hash=target_hash,
        )

        adv_image = torch.clamp(adv_image, clamp_min, clamp_max).detach()

        for i in range(num_iter):
            adv_image = adv_image.detach().requires_grad_(True)

            if self.args.model == "DNPH":
                image_hash, _ = self.model.encode_image(adv_image)
            else:
                image_hash = self.model.encode_image(adv_image)

            self.model.clip.zero_grad()
            self.model.zero_grad()
            self.model2.zero_grad()
            loss, loss_holistic, loss_instance = self._multi_scale_hamming_alignment_loss(
                adv_hash=image_hash,
                holistic_target_hash=target_hash,
                reliable_anchors=reliable_anchors,
            )

            self.writer.add_scalar("TotalLoss", loss, i)
            self.writer.add_scalar("HIMA/Holistic", loss_holistic, i)
            self.writer.add_scalar("HIMA/InstanceAnchor", loss_instance, i)

            loss.backward()

            with torch.no_grad():
                perturb = alpha * adv_image.grad.sign()
                adv_image = adv_image - perturb
                adv_image = torch.min(torch.max(adv_image, orig_image - epsilon), orig_image + epsilon)
                adv_image = torch.clamp(adv_image, clamp_min, clamp_max)

            if i % 20 == 0 and i <= 140:
                alpha = alpha / 2
            if i == 1800:
                alpha = alpha / 2

        return adv_image

    def text_attack(
        self,
        text: Union[torch.Tensor, List[str]],
        target_image,
        query_labels,
        index,
        target_features: torch.Tensor,
        origin_features: torch.Tensor,
        target_hash: torch.Tensor,
        origin_hash: torch.Tensor,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 100,
        clamp_min: float = None,
        clamp_max: float = None,
    ) -> torch.Tensor:
        self.model.clip.eval()
        self.model.eval()
        self.model2.eval()

        if isinstance(text, list) or isinstance(text, str):
            text_input = self._tokenize_text(text)
        elif text.dtype == torch.long:
            text_input = self._tokens_to_embedding(text.clone().detach().to(self.rank))
        else:
            text_input = text.to(torch.float32).to(self.rank)

        text_input = text_input.float()
        orig_embed = text_input.clone().detach()

        if clamp_min is None:
            min_val = orig_embed.min().item() - 5
            max_val = orig_embed.max().item() + 5
            clamp_min = min(min_val, -5.0)
            clamp_max = max(max_val, 5.0)

        adv_embed = orig_embed.clone().detach().requires_grad_(True)
        device = adv_embed.device
        target_hash = target_hash.detach().to(device)
        reliable_anchors = self._target_image_instance_anchors(
            target_image=target_image,
            text=text,
            target_labels=query_labels,
            index=index,
            holistic_target_hash=target_hash,
        )

        original_alpha = alpha

        for i in range(num_iter):
            if adv_embed.grad is not None:
                adv_embed.grad.zero_()

            text_hash = self._get_text_hash(adv_embed)
            text_instance_hashes = self._get_text_instance_hashes(adv_embed)

            total_loss, loss_holistic, loss_instance, loss_text_instance = self._text_holistic_instance_loss(
                adv_text_hash=text_hash,
                adv_text_instance_hashes=text_instance_hashes,
                holistic_target_hash=target_hash,
                reliable_anchors=reliable_anchors,
            )

            self.writer.add_scalar("TotalLoss", total_loss, i)
            self.writer.add_scalar("HIMA/Holistic", loss_holistic, i)
            self.writer.add_scalar("HIMA/InstanceAnchor", loss_instance, i)
            self.writer.add_scalar("HIMA/TextInstanceAnchor", loss_text_instance, i)

            try:
                total_loss.backward()
            except RuntimeError as e:
                self.logger.error(f"Backward iteration {i + 1}/{num_iter} failed: {str(e)}")
                adv_embed = adv_embed.detach().requires_grad_(True)
                continue

            with torch.no_grad():
                if adv_embed.grad is not None:
                    perturb = alpha * adv_embed.grad.sign()
                    adv_embed -= perturb
                    adv_embed = torch.min(torch.max(adv_embed, orig_embed - epsilon), orig_embed + epsilon)
                    adv_embed = torch.clamp(adv_embed, clamp_min, clamp_max)
                    adv_embed = adv_embed.detach().requires_grad_(True)
                    if i % 20 == 0:
                        alpha = max(original_alpha * 0.5, alpha * 0.8)
                else:
                    noise = torch.randn_like(adv_embed) * 0.01 * epsilon
                    adv_embed += noise
                    adv_embed = adv_embed.detach().requires_grad_(True)

        return adv_embed

    def _tokenize_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        if not hasattr(self, "clip_tokenizer"):
            from attacked.models.CLIP import clip

            self.clip_tokenizer = clip.tokenize

        texts = [text] if isinstance(text, str) else text
        text_tokens = self.clip_tokenizer(texts, truncate=True).to(self.rank)
        return self._tokens_to_embedding(text_tokens)

    def _tokens_to_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed = self.model.clip.token_embedding(token_ids)
        seq_len = token_ids.size(1)
        if hasattr(self.model.clip, "positional_embedding") and self.model.clip.positional_embedding is not None:
            pos_embed = self.model.clip.positional_embedding[:seq_len].unsqueeze(0)
            embed += pos_embed.to(embed.device)
        return embed.float()

    def _get_text_token_features(self, pure_embed: torch.Tensor) -> torch.Tensor:
        x = pure_embed.permute(1, 0, 2)

        if hasattr(self.model.clip.transformer, "forward"):
            x, _ = self.model.clip.transformer(x, key_padding_mask=None)
        else:
            for layer in self.model.clip.transformer:
                x = layer(x)

        x = x.permute(1, 0, 2)

        if hasattr(self.model.clip, "ln_final"):
            x = self.model.clip.ln_final(x)

        if hasattr(self.model.clip, "text_projection") and self.model.clip.text_projection is not None:
            x = x @ self.model.clip.text_projection

        return x

    def _resize_text_tokens_for_instance_probe(self, token_features: torch.Tensor) -> torch.Tensor:
        expected_tokens = getattr(
            self.model2.instance_hash_projector.txt_token_hash.token_layer,
            "in_channels",
            token_features.shape[1],
        )
        current_tokens = token_features.shape[1]

        if current_tokens > expected_tokens:
            return token_features[:, :expected_tokens, :]
        if current_tokens < expected_tokens:
            pad = token_features[:, -1:, :].expand(-1, expected_tokens - current_tokens, -1)
            return torch.cat([token_features, pad], dim=1)
        return token_features

    def _get_text_instance_hashes(self, adv_embed: torch.Tensor) -> torch.Tensor:
        token_features = self._get_text_token_features(adv_embed)
        token_features = self._resize_text_tokens_for_instance_probe(token_features)
        token_features = token_features.to(dtype=next(self.model2.parameters()).dtype)
        instance_hashes, _ = self.model2.instance_hash_projector.encode_txt(token_features)
        return instance_hashes.float()

    def _get_text_hash(self, adv_embed: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = adv_embed.shape
        x = adv_embed.permute(1, 0, 2)

        if hasattr(self.model.clip.transformer, "forward"):
            x, _ = self.model.clip.transformer(x, key_padding_mask=None)
        else:
            for layer in self.model.clip.transformer:
                x = layer(x)

        x = x.permute(1, 0, 2)

        if hasattr(self.model.clip, "ln_final"):
            x = self.model.clip.ln_final(x)

        eos_pos = seq_len - 1
        text_features = x[:, eos_pos]
        text_hash = self.model.encode_text(text_features, 1)

        if self.args.model == "DNPH":
            if isinstance(text_hash, tuple):
                return text_hash[0]
            return text_hash
        return text_hash

    def attack(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img, query_txt = self.get_attack_code(self.query_loader, None, self.args.query_num)
        target_img, target_txt = self.get_attack_code(self.target_loader, None, self.args.query_num)
        query_advimg, query_advtxt = self.get_attack_code(self.query_loader, self.target_loader, self.args.query_num)
        retrieval_img, retrieval_txt = self.get_attack_code(self.retrieval_loader, None, self.args.retrieval_num)

        if self.args.is_target:
            adv_mAPi2t = calc_soft_map_k(query_advimg, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)
            adv_mAPt2i = calc_soft_map_k(query_advtxt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            adv_mAPi2i = calc_soft_map_k(query_advimg, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            adv_mAPt2t = calc_soft_map_k(query_advtxt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)

            tar_mAPi2t = calc_soft_map_k(target_img, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)
            tar_mAPt2i = calc_soft_map_k(target_txt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            tar_mAPi2i = calc_soft_map_k(target_img, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            tar_mAPt2t = calc_soft_map_k(target_txt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)

            mAPi2t = calc_soft_map_k(query_img, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)
            mAPt2i = calc_soft_map_k(query_txt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            mAPi2i = calc_soft_map_k(query_img, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            mAPt2t = calc_soft_map_k(query_txt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)
        else:
            mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, self.args.k, self.rank)
            mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, self.args.k, self.rank)
            mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, self.args.k, self.rank)
            mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, self.args.k, self.rank)

        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f"pretrain:{self.args.pretrained}")
        self.logger.info(f"attack:{self.args.is_attack}, target:{self.args.is_target}, k:{self.args.k}")
        self.logger.info(f"epsilon={self.args.epsilon}, alpha={self.args.alpha}, num_iter={self.args.num_iter}")

        if self.args.is_target:
            self.logger.info(f">>>>>>Adv-MAP(i->t): {adv_mAPi2t}, MAP(t->i): {adv_mAPt2i}, MAP(t->t): {adv_mAPt2t}, MAP(i->i): {adv_mAPi2i}")
            self.logger.info(f">>>>>>Tar-MAP(i->t): {tar_mAPi2t}, MAP(t->i): {tar_mAPt2i}, MAP(t->t): {tar_mAPt2t}, MAP(i->i): {tar_mAPi2i}")
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

    def get_attack_code(self, data_loader, target_loader, length: int):
        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        num = 0

        if target_loader is not None:
            for query, target in tqdm(zip(data_loader, target_loader)):
                image = query[0].to(self.rank, non_blocking=True)
                text = query[1].to(self.rank, non_blocking=True)
                index = query[3].to(self.rank)

                target_image = target[0].to(self.rank, non_blocking=True)
                target_text = target[1].to(self.rank, non_blocking=True)
                target_labels = target[2].to(self.rank)
                target_index = target[3].numpy()

                if self.args.is_image:
                    origin_features = self.model.clip.encode_text(text).detach()
                    target_features = self.model.clip.encode_text(target_text).detach()
                    if self.args.model == "DNPH":
                        origin_hash, _ = self.model.encode_text(text)
                        target_hash, _ = self.model.encode_text(target_text)
                    else:
                        origin_hash = self.model.encode_text(text)
                        target_hash = self.model.encode_text(target_text)

                if self.args.is_text:
                    origin_features = self.model.clip.encode_image(image).detach()
                    target_features = self.model.clip.encode_image(target_image).detach()
                    if self.args.model == "DNPH":
                        origin_hash, _ = self.model.encode_image(image)
                        target_hash, _ = self.model.encode_image(target_image)
                    else:
                        origin_hash = self.model.encode_image(image)
                        target_hash = self.model.encode_image(target_image)

                if self.args.is_image:
                    adv_image = self.image_attack(
                        num,
                        image=image,
                        target_text=target_text,
                        query_labels=target_labels,
                        index=index,
                        target_features=target_features,
                        origin_features=origin_features,
                        target_hash=target_hash,
                        origin_hash=origin_hash,
                        epsilon=self.args.epsilon,
                        alpha=self.args.alpha,
                        num_iter=self.args.num_iter,
                    )
                    num = num + 1
                    if self.args.model == "DNPH":
                        image_hash, _ = self.model.encode_image(adv_image)
                        text_hash, _ = self.model.encode_text(text)
                    else:
                        image_hash = self.model.encode_image(adv_image)
                        text_hash = self.model.encode_text(text)

                if self.args.is_text:
                    adv_text = self.text_attack(
                        text=text,
                        target_image=target_image,
                        query_labels=target_labels,
                        index=index,
                        target_features=target_features,
                        origin_features=origin_features,
                        target_hash=target_hash,
                        origin_hash=origin_hash,
                        epsilon=self.args.epsilon,
                        alpha=self.args.alpha,
                        num_iter=self.args.num_iter,
                    )
                    if self.args.model == "DNPH":
                        image_hash, _ = self.model.encode_image(image)
                        text_hash, _ = self.model.encode_text(adv_text, 1)
                    else:
                        image_hash = self.model.encode_image(image)
                        text_hash = self.model.encode_text(adv_text, 1)

                if self.args.model == "DCMHT":
                    img_buffer[target_index, :] = image_hash.data.sign_()
                    text_buffer[target_index, :] = text_hash.data.sign_()
                else:
                    img_buffer[target_index, :] = image_hash.data
                    text_buffer[target_index, :] = text_hash.data
        else:
            for image, text, label, index in tqdm(data_loader):
                image = image.to(self.rank, non_blocking=True)
                text = text.to(self.rank, non_blocking=True)
                index = index.numpy()

                if self.args.model == "DNPH":
                    image_hash, _ = self.model.encode_image(image)
                    text_hash, _ = self.model.encode_text(text)
                else:
                    image_hash = self.model.encode_image(image)
                    text_hash = self.model.encode_text(text)

                if self.args.model == "DCMHT":
                    img_buffer[index, :] = image_hash.data.sign_()
                    text_buffer[index, :] = text_hash.data.sign_()
                else:
                    img_buffer[index, :] = image_hash.data
                    text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer

def read_config(config_file):
    assert os.path.isfile(config_file), f"config file {config_file} doesn't eixst!"
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_file)
    except ModuleNotFoundError:
        import yaml

        class ConfigNode(dict):
            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError as exc:
                    raise AttributeError(item) from exc

        def wrap(value):
            if isinstance(value, dict):
                return ConfigNode({key: wrap(item) for key, item in value.items()})
            if isinstance(value, list):
                return [wrap(item) for item in value]
            return value

        with open(config_file, "r") as f:
            cfg = wrap(yaml.safe_load(f))
    return cfg
