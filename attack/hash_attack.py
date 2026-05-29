import json
from datetime import datetime
from typing import Union, List

from model.IHIR import IHIR

import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio

import torch.nn.functional as F

from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args
from utils.calc_utils import calc_map_k_matrix as calc_map_k, \
    visualize_cross_modal_similarity, target_drift_statistics, calc_TDG, calc_Drift, calc_soft_map_k
from dataset.dataloader import dataloader

class Trainer(TrainBase):

    def __init__(self,
                rank=0):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.logger.info("ViT+GPT!")

        if self.args.model == 'DCMHT':
            from attacked.models.DCMHT import DCMHT
            HashModel = DCMHT
            cfg = read_config("attacked/configs/DCMHT/config.yaml")
            self.model = HashModel(cfg, outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)
            cfg2 = read_config("attacked/configs/IHIR/config.yaml")
            self.model2 = IHIR.from_config(cfg2, output_dim=self.args.output_dim, text_token_count=self.args.max_words).to(self.rank)

        if self.args.model == 'DNPH':
            from attacked.models.DNPH import DNPH
            HashModel = DNPH
            cfg = read_config("attacked/configs/DNPH/config.yaml")
            self.model = HashModel(cfg,outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)
            cfg2 = read_config("attacked/configs/IHIR/config.yaml")
            self.model2 = IHIR.from_config(cfg2, output_dim=self.args.output_dim, text_token_count=self.args.max_words).to(self.rank)
            # HashModel3 = DSPH
            # cfg3 = read_config("attacked/configs/DSPH/config.yaml")
            # self.model3 = HashModel3(cfg3, outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)

        if self.args.model == 'DSPH':
            from attacked.models.DSPH import DSPH
            HashModel = DSPH
            cfg = read_config("attacked/configs/DSPH/config.yaml")
            self.model = HashModel(cfg, outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)
            cfg2 = read_config("attacked/configs/IHIR/config.yaml")
            self.model2 = IHIR.from_config(cfg2, output_dim=self.args.output_dim, text_token_count=self.args.max_words).to(self.rank)
            # HashModel3 = DNPH
            # cfg3 = read_config("attacked/configs/DNPH/config.yaml")
            # self.model3 = HashModel3(cfg3, outputDim=self.args.output_dim, clipPath=self.args.clip_path).to(self.rank)

        if self.args.model == 'MITH':
            raise RuntimeError("MITH is not available in this repository.")

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
            self.model2.load_state_dict(torch.load(self.args.pretrained2, map_location=f"cuda:{self.rank}"))
            # self.model3.load_state_dict(torch.load(self.args.pretrained3, map_location=f"cuda:{self.rank}"))

        self.model.float()
        self.model2.float()
        self.model2.eval()
        for param in self.model2.parameters():
            param.requires_grad_(False)
        # self.model3.float()
        # DCMHT
        # self.optimizer = BertAdam([
        #             {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
        #             {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
        #             {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
        #             ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
        #             b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
        #             weight_decay=self.args.weight_decay, max_grad_norm=1.0)
        #DNPH DSPH
        self.optimizer = BertAdam([
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.hash.parameters(), 'lr': self.args.lr},
            ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)
        # print(self.model)

        # self.distance = SetwiseDistance(img_set_size=32, txt_set_size=32,
        #                                 denominator=2.0,
        #                                 temperature=16.0,
        #                                 temperature_txt_scale=1.0,
        #                                 mode="chamfer")

        # self.triplet_loss = TripletLoss(reduction="mean")

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.ref_model = BertForMaskedLM.from_pretrained('bert-base-uncased', use_safetensors=False)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data, target_data= dataloader(captionFile=self.args.caption_file,
                                        indexFile=self.args.index_file, 
                                        labelFile=self.args.label_file, 
                                        maxWords=self.args.max_words,
                                        imageResolution=self.args.resolution,
                                        query_num=self.args.query_num,
                                        train_num=self.args.train_num,
                                        seed=self.args.seed)
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
                shuffle=True
            )
        self.query_loader = DataLoader(
                dataset=query_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.target_loader = DataLoader(
            dataset=target_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )


    def _hima_arg(self, name: str, default):
        return getattr(self.args, name, default)

    def _hima_soft_binary(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(h)

    def _hima_match_dim(self, a: torch.Tensor, b: torch.Tensor):
        dim = min(a.shape[-1], b.shape[-1])
        return a[..., :dim], b[..., :dim]

    def _hima_hamming_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Normalized differentiable Hamming distance:
            d_H(u, v) = (L - u^T v) / 2L.
        Supports [B, L] against [B, L] or [B, K, L].
        """
        a, b = self._hima_match_dim(a, b)
        if b.dim() == 3 and a.dim() == 2:
            a = a.unsqueeze(1)
        return 0.5 * (1.0 - (a * b).mean(dim=-1))

    def _hima_select_reliable_anchors(
            self,
            instance_hashes: torch.Tensor,
            holistic_target_hash: torch.Tensor,
            top_k: int = None,
    ) -> torch.Tensor:
        """
        Reliability-aware instance anchor screening.

        The paper defines reliability through consistency with the holistic
        target direction under uncertainty. In this codebase, model2 already
        exposes token/instance-level hashes, so we use a differentiable proxy:
        Hamming affinity to the holistic target plus a small confidence term
        from binary saturation. The selected anchors are detached and treated
        as fixed target guidance during PGD, matching HIMA.
        """
        if instance_hashes.dim() == 2:
            instance_hashes = instance_hashes.unsqueeze(1)

        top_k = top_k or self._hima_arg("hima_topk", self._hima_arg("token", 4))
        top_k = max(1, min(int(top_k), instance_hashes.shape[1]))

        soft_instances = self._hima_soft_binary(instance_hashes.detach())
        soft_target = self._hima_soft_binary(holistic_target_hash.detach()).unsqueeze(1)
        soft_instances, soft_target = self._hima_match_dim(soft_instances, soft_target)

        affinity = (soft_instances * soft_target).mean(dim=-1)
        confidence = soft_instances.abs().mean(dim=-1)
        reliability = affinity + 0.1 * confidence

        topk_idx = torch.topk(reliability, k=top_k, dim=1).indices
        batch_idx = torch.arange(instance_hashes.shape[0], device=instance_hashes.device).unsqueeze(1)
        return instance_hashes[batch_idx, topk_idx].detach()

    def _hima_multiscale_loss(
            self,
            adv_hash: torch.Tensor,
            holistic_target_hash: torch.Tensor,
            reliable_anchors: torch.Tensor,
    ):
        """
        HIMA objective:
            L_total = L_holistic + lambda * L_instance_anchor.
        """
        lam = float(self._hima_arg("hima_lambda", 1.0))
        margin = float(self._hima_arg("hima_margin", 0.1))

        soft_adv = self._hima_soft_binary(adv_hash)
        soft_target = self._hima_soft_binary(holistic_target_hash.detach())
        soft_anchors = self._hima_soft_binary(reliable_anchors.detach())

        loss_holistic = self._hima_hamming_distance(soft_adv, soft_target).mean()

        anchor_dist = self._hima_hamming_distance(soft_adv, soft_anchors)
        weakest_dist = anchor_dist.max(dim=1, keepdim=True).values.detach()
        loss_instance = F.relu(anchor_dist - weakest_dist + margin).mean()

        loss_total = loss_holistic + lam * loss_instance
        return loss_total, loss_holistic.detach(), loss_instance.detach()

    def _hima_instance_set_anchor_loss(
            self,
            adv_instance_hashes: torch.Tensor,
            reliable_anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Align adversarial text instances with reliable target anchors.
        This is the text-side counterpart of HIMA's instance-level guidance.
        """
        if adv_instance_hashes.dim() == 2:
            adv_instance_hashes = adv_instance_hashes.unsqueeze(1)
        if reliable_anchors.dim() == 2:
            reliable_anchors = reliable_anchors.unsqueeze(1)

        soft_adv_instances = self._hima_soft_binary(adv_instance_hashes)
        soft_anchors = self._hima_soft_binary(reliable_anchors.detach())
        soft_adv_instances, soft_anchors = self._hima_match_dim(soft_adv_instances, soft_anchors)

        # [B, M, K, L] -> [B, M, K]
        pairwise_dist = 0.5 * (
            1.0 - (soft_adv_instances.unsqueeze(2) * soft_anchors.unsqueeze(1)).mean(dim=-1)
        )
        return pairwise_dist.min(dim=2).values.mean()

    def _hima_text_multiscale_loss(
            self,
            adv_text_hash: torch.Tensor,
            adv_text_instance_hashes: torch.Tensor,
            holistic_target_hash: torch.Tensor,
            reliable_anchors: torch.Tensor,
    ):
        """
        Text-attack HIMA objective.

        It keeps the victim text hash aligned with the holistic target image
        hash and reliable target image anchors, while also aligning the
        adversarial text instance hashes with those anchors.
        """
        loss_total, loss_holistic, loss_anchor = self._hima_multiscale_loss(
            adv_hash=adv_text_hash,
            holistic_target_hash=holistic_target_hash,
            reliable_anchors=reliable_anchors,
        )
        text_instance_weight = float(self._hima_arg("hima_text_instance_lambda", 0.5))
        loss_text_instance = self._hima_instance_set_anchor_loss(
            adv_instance_hashes=adv_text_instance_hashes,
            reliable_anchors=reliable_anchors,
        )
        loss_total = loss_total + text_instance_weight * loss_text_instance
        return loss_total, loss_holistic, loss_anchor, loss_text_instance.detach()

    def _hima_target_text_anchors(
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
        return self._hima_select_reliable_anchors(target_text_instances, holistic_target_hash)

    def _hima_target_image_anchors(
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
        return self._hima_select_reliable_anchors(target_image_instances, holistic_target_hash)
    #     mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
    #     self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt = self.get_code(self.query_loader, self.args.query_num) if self.args.hash_layer == "select" else super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = self.get_code(self.retrieval_loader, self.args.retrieval_num) if self.args.hash_layer == "select" else super().get_code(self.retrieval_loader, self.args.retrieval_num)
        # print("get all code")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
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
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")

    def image_attack(
        self,
        num,
        image: torch.Tensor,
        target_text: torch.Tensor, query_labels, index,
        target_features,
        origin_features,
        target_hash,
        origin_hash,
        epsilon,  # Perturbation budget (L-infinity constraint)
        alpha,   # Single-step attack size
        num_iter,    # Number of iterations
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
        reliable_anchors = self._hima_target_text_anchors(
            image=orig_image,
            target_text=target_text,
            target_labels=query_labels,
            index=index,
            holistic_target_hash=target_hash,
        )

        # Initialize random perturbation
        adv_image = adv_image + torch.zeros_like(adv_image)
        adv_image = torch.clamp(adv_image, clamp_min, clamp_max).detach()

        for i in range(num_iter):

            # adv_image.requires_grad = True
            adv_image = adv_image.detach().requires_grad_(True)

            #DNPH
            if self.args.model == "DNPH":
                image_hash,_= self.model.encode_image(adv_image)
            #DSPH
            else:
                image_hash= self.model.encode_image(adv_image)

            # if self.args.hash_layer == "select":
            #     image_hash = torch.cat(image_hash, dim=-1) if isinstance(image_hash, list) else image_hash.view(image_hash.shape[0], -1)

            self.model.clip.zero_grad()
            self.model.zero_grad()
            self.model2.zero_grad()
            loss, loss_holistic, loss_instance = self._hima_multiscale_loss(
                adv_hash=image_hash,
                holistic_target_hash=target_hash,
                reliable_anchors=reliable_anchors,
            )

            self.writer.add_scalar("TotalLoss", loss, i)
            self.writer.add_scalar("HIMA/Holistic", loss_holistic, i)
            self.writer.add_scalar("HIMA/InstanceAnchor", loss_instance, i)


            # Gradient computation
            loss.backward()

            with torch.no_grad():
                perturb = alpha * adv_image.grad.sign()
                adv_image = adv_image - perturb
                adv_image = torch.min(torch.max(adv_image, orig_image - epsilon), orig_image + epsilon)
                adv_image = torch.clamp(adv_image, clamp_min, clamp_max)

            if i % 20 == 0 and i <= 140:
                alpha = alpha / 2
            if i == 1800 : alpha = alpha / 2
            #
            # if i % 50 == 0 and i <= 200:
            #     alpha = alpha / 2

        return adv_image

    def text_attack(
            self,
            text: Union[torch.Tensor, List[str]],  # Supports raw text or token IDs
            target_image,query_labels, index,
            target_features: torch.Tensor,
            origin_features: torch.Tensor,
            target_hash: torch.Tensor,
            origin_hash: torch.Tensor,
            epsilon: float = 0.03,
            alpha: float = 0.01,
            num_iter: int = 100,
            clamp_min: float = None,
            clamp_max: float = None,
    ) -> torch.Tensor:  # Returns Long token IDs

        # Ensure models are in evaluation mode
        self.model.clip.eval()
        self.model.eval()
        self.model2.eval()

        # ==================== Input preprocessing ====================
        if isinstance(text, list) or isinstance(text, str):
            # Handle raw text input
            text_input = self._tokenize_text(text)
        elif text.dtype == torch.long:
            # Already token IDs - store for later conversion
            original_tokens = text.clone().detach().to(self.rank)
            text_input = self._tokens_to_embedding(original_tokens)
        else:
            # Floating-point embeddings - use directly
            text_input = text.to(torch.float32).to(self.rank)
            original_tokens = None

        # Ensure input is floating point
        text_input = text_input.float()

        # ==================== Initialize adversarial embeddings ====================
        orig_embed = text_input.clone().detach()

        # Automatically compute reasonable value bounds
        if clamp_min is None:
            min_val = orig_embed.min().item() - 5
            max_val = orig_embed.max().item() + 5
            clamp_min = min(min_val, -5.0)
            clamp_max = max(max_val, 5.0)

        # Create optimizable adversarial embeddings
        adv_embed = orig_embed.clone().detach().requires_grad_(True)

        # Move hash guidance to the corresponding device
        device = adv_embed.device
        target_hash = target_hash.detach().to(device)
        reliable_anchors = self._hima_target_image_anchors(
            target_image=target_image,
            text=text,
            target_labels=query_labels,
            index=index,
            holistic_target_hash=target_hash,
        )

        # Save the original alpha value
        original_alpha = alpha

        # ==================== Optimization loop ====================
        for i in range(num_iter):
            if adv_embed.grad is not None:
                adv_embed.grad.zero_()

            # Extract hash features
            text_hash = self._get_text_hash(adv_embed)
            text_instance_hashes = self._get_hima_text_instance_hashes(adv_embed)

            total_loss, loss_holistic, loss_instance, loss_text_instance = self._hima_text_multiscale_loss(
                adv_text_hash=text_hash,
                adv_text_instance_hashes=text_instance_hashes,
                holistic_target_hash=target_hash,
                reliable_anchors=reliable_anchors,
            )

            self.writer.add_scalar("TotalLoss", total_loss, i)
            self.writer.add_scalar("HIMA/Holistic", loss_holistic, i)
            self.writer.add_scalar("HIMA/InstanceAnchor", loss_instance, i)
            self.writer.add_scalar("HIMA/TextInstanceAnchor", loss_text_instance, i)

            # Backpropagation
            try:
                total_loss.backward()
            except RuntimeError as e:
                # Create a new variable and continue running
                self.logger.error(f"Backward iteration {i + 1}/{num_iter} failed: {str(e)}")
                adv_embed = adv_embed.detach().requires_grad_(True)
                continue

            # Gradient update
            with torch.no_grad():
                if adv_embed.grad is not None:
                    # 1. Apply gradients
                    grad = adv_embed.grad
                    perturb = alpha * grad.sign()
                    adv_embed -= perturb

                    adv_embed = torch.min(torch.max(adv_embed, orig_embed - epsilon), orig_embed + epsilon)
                    adv_embed = torch.clamp(adv_embed, clamp_min, clamp_max)
                    # 4. Reset gradient state
                    adv_embed = adv_embed.detach().requires_grad_(True)

                    # Dynamically adjust the learning rate
                    if i % 20 == 0:
                        alpha = max(original_alpha * 0.5, alpha * 0.8)
                else:
                    # Handle None gradients
                    noise = torch.randn_like(adv_embed) * 0.01 * epsilon
                    adv_embed += noise
                    adv_embed = adv_embed.detach().requires_grad_(True)
        # print(orig_embed)
        # print(adv_embed)

        return adv_embed

    # ==================== Helper functions ====================
    def _tokenize_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Convert text to embedding vectors."""
        # Ensure the tokenizer is globally available
        if not hasattr(self, 'clip_tokenizer'):
            from attacked.models.CLIP import clip
            self.clip_tokenizer = clip.tokenize

        # Handle multiple input formats
        texts = [text] if isinstance(text, str) else text

        # Tokenize
        text_tokens = self.clip_tokenizer(texts, truncate=True).to(self.rank)

        # Convert to embedding vectors
        return self._tokens_to_embedding(text_tokens)

    def _tokens_to_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embedding vectors, including positional embeddings."""
        # Base embeddings
        embed = self.model.clip.token_embedding(token_ids)

        # Add positional embeddings
        seq_len = token_ids.size(1)
        if hasattr(self.model.clip, 'positional_embedding') and self.model.clip.positional_embedding is not None:
            pos_embed = self.model.clip.positional_embedding[:seq_len].unsqueeze(0)
            embed += pos_embed.to(embed.device)

        return embed.float()

    def _get_text_token_features(self, pure_embed: torch.Tensor) -> torch.Tensor:
        """Extract differentiable token-level text features from adversarial embeddings."""
        x = pure_embed.permute(1, 0, 2)

        if hasattr(self.model.clip.transformer, 'forward'):
            x, _ = self.model.clip.transformer(x, key_padding_mask=None)
        else:
            for layer in self.model.clip.transformer:
                x = layer(x)

        x = x.permute(1, 0, 2)

        if hasattr(self.model.clip, 'ln_final'):
            x = self.model.clip.ln_final(x)

        if hasattr(self.model.clip, 'text_projection') and self.model.clip.text_projection is not None:
            x = x @ self.model.clip.text_projection

        return x

    def _resize_text_tokens_for_hima_probe(self, token_features: torch.Tensor) -> torch.Tensor:
        """Match adversarial text token count to the auxiliary instance probe."""
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

    def _get_hima_text_instance_hashes(self, adv_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable adversarial text instance hashes with the frozen
        auxiliary instance probe.
        """
        token_features = self._get_text_token_features(adv_embed)
        token_features = self._resize_text_tokens_for_hima_probe(token_features)
        token_features = token_features.to(dtype=next(self.model2.parameters()).dtype)
        instance_hashes, _ = self.model2.instance_hash_projector.encode_txt(token_features)
        return instance_hashes.float()

    def _get_text_hash(self, adv_embed: torch.Tensor) -> torch.Tensor:
        """Differentiable text hash feature extraction."""
        # Directly use embedding vectors to compute hash features
        batch_size, seq_len, embed_dim = adv_embed.shape

        # 1. Pass through Transformer while preserving differentiability
        x = adv_embed.permute(1, 0, 2)  # (seq_len, batch, dim)

        if hasattr(self.model.clip.transformer, 'forward'):
            x, _ = self.model.clip.transformer(x, key_padding_mask=None)
        else:
            for layer in self.model.clip.transformer:
                x = layer(x)

        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)

        # 2. Final layer normalization
        if hasattr(self.model.clip, 'ln_final'):
            x = self.model.clip.ln_final(x)

        # 3. Extract features at the EOS position
        eos_pos = seq_len - 1
        text_features = x[:, eos_pos]

         # Assume a dedicated hash layer exists
        text_hash = self.model.encode_text(text_features,1)

        # Process output according to model type
        if self.args.model == "DNPH":
            if isinstance(text_hash, tuple):
                return text_hash[0]  # Return the primary hash features
            return text_hash
        return text_hash

    def attack(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        # Query hash codes without attack
        query_img, query_txt = self.get_attack_code(self.query_loader, None,self.args.query_num)

        target_img, target_txt = self.get_attack_code(self.target_loader, None,self.args.query_num)

        # Pass target during attack and return attacked query hash codes
        query_advimg, query_advtxt= self.get_attack_code(self.query_loader, self.target_loader,self.args.query_num)

        retrieval_img, retrieval_txt = self.get_attack_code(self.retrieval_loader, None,self.args.retrieval_num)

        if self.args.is_target:

            drift_gaps = []

            for i in range(len(query_advimg)):
                stat = target_drift_statistics(
                    adv_hash=query_advimg[i],
                    target_hash=target_txt[i],
                    retrieval_hash=retrieval_txt,
                    retrieval_labels=self.retrieval_labels,
                    target_label=self.target_labels[i]
                )
                drift_gaps.append(stat["drift_gap"])

            print("Mean Target Drift Gap:", sum(drift_gaps) / len(drift_gaps))

            visualize_cross_modal_similarity(query_txt, query_img, self.query_labels, self.query_labels, "qtqi")
            visualize_cross_modal_similarity(query_txt, query_advimg, self.query_labels, self.query_labels, "qtqai")
            visualize_cross_modal_similarity(query_txt, target_img, self.query_labels, self.query_labels, "qtti")
            visualize_cross_modal_similarity(target_txt, target_img, self.target_labels, self.target_labels, "ttti")
            visualize_cross_modal_similarity(target_txt, query_img, self.target_labels, self.target_labels,"ttqi")
            visualize_cross_modal_similarity(target_txt, query_advimg, self.target_labels, self.target_labels,"ttqai")


            # visualize_cross_modal_similarity(query_img, query_txt, self.query_labels, self.query_labels, "qiqt")
            # visualize_cross_modal_similarity(query_img, query_advtxt, self.query_labels, self.query_labels, "qiqat")
            # visualize_cross_modal_similarity(query_img, target_txt, self.query_labels, self.query_labels, "qitt")
            # visualize_cross_modal_similarity(target_img, target_txt, self.target_labels, self.target_labels, "titt")
            # visualize_cross_modal_similarity(target_img, query_txt, self.target_labels, self.target_labels,"tiqt")
            # visualize_cross_modal_similarity(target_img, query_advtxt, self.target_labels, self.target_labels,"tiqat")



            TDG = calc_TDG(query_advimg,retrieval_txt,self.target_labels,self.retrieval_labels)
            Drift = calc_Drift(query_img,query_advimg,retrieval_txt,self.retrieval_labels,self.target_labels)

            print(f"Mean TDG ↓ : {TDG.item():.4f}")
            print(f"Mean Drift ↓ : {Drift.item():.4f}")

            # target_img, target_txt = self.get_attack_code(self.target_loader, None, self.args.query_num)
            # In targeted attack mode, compute attacked and clean mAP under the same target
            # adv_mAPi2t = calc_map_k(query_advimg, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)
            # adv_mAPt2i = calc_map_k(query_advtxt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)
            # adv_mAPi2i = calc_map_k(query_advimg, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)
            # adv_mAPt2t = calc_map_k(query_advtxt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)

            adv_mAPi2t = calc_soft_map_k(query_advimg, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)
            adv_mAPt2i = calc_soft_map_k(query_advtxt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            adv_mAPi2i = calc_soft_map_k(query_advimg, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            adv_mAPt2t = calc_soft_map_k(query_advtxt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)

            # tar_mAPi2t = calc_map_k(target_img, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)
            # tar_mAPt2i = calc_map_k(target_txt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)
            # tar_mAPi2i = calc_map_k(target_img, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)
            # tar_mAPt2t = calc_map_k(target_txt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k,
            #                         self.rank)

            tar_mAPi2t = calc_soft_map_k(target_img, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)
            tar_mAPt2i = calc_soft_map_k(target_txt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            tar_mAPi2i = calc_soft_map_k(target_img, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            tar_mAPt2t = calc_soft_map_k(target_txt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)

            # mAPi2t = calc_map_k(query_img, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k,
            #                     self.rank)
            # mAPt2i = calc_map_k(query_txt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k,
            #                     self.rank)
            # mAPi2i = calc_map_k(query_img, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k,
            #                     self.rank)
            # mAPt2t = calc_map_k(query_txt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k,
            #                     self.rank)

            mAPi2t = calc_soft_map_k(query_img, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)
            mAPt2i = calc_soft_map_k(query_txt, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            mAPi2i = calc_soft_map_k(query_img, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k)
            mAPt2t = calc_soft_map_k(query_txt, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k)


        else:
            # if self.args.is_attack:
            #     adv_mAPi2t = calc_map_k(query_advimg, retrieval_txt, self.target_labels, self.retrieval_labels, self.args.k, self.rank)
            #     adv_mAPt2i = calc_map_k(query_advtxt, retrieval_img, self.target_labels, self.retrieval_labels,self.args.k, self.rank)
            #     adv_mAPi2i = calc_map_k(query_advimg, retrieval_img, self.target_labels, self.retrieval_labels, self.args.k, self.rank)
            #     adv_mAPt2t = calc_map_k(query_advtxt, retrieval_txt, self.target_labels, self.retrieval_labels,self.args.k, self.rank)
            #
            #     mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, self.args.k, self.rank)
            #     mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, self.args.k, self.rank)
            #     mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, self.args.k, self.rank)
            #     mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, self.args.k, self.rank)
            # else:
            mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, self.args.k,self.rank)
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
        # Targeted mode attacks by default; untargeted mode distinguishes attacked and clean cases
        # In targeted attack mode, set attack to false when computing original mAP
        if target_loader is not None:
            for query, target in tqdm(zip(data_loader, target_loader)):
                image = query[0].to(self.rank, non_blocking=True)
                text = query[1].to(self.rank, non_blocking=True)
                index = query[3].to(self.rank)

                target_image = target[0].to(self.rank, non_blocking=True)
                target_text = target[1].to(self.rank, non_blocking=True)
                target_labels = target[2].to(self.rank)
                target_index = target[3]

                image = image.to(self.rank, non_blocking=True)
                text = text.to(self.rank, non_blocking=True)


                # index = index.numpy()
                target_index = target_index.numpy()

                if self.args.is_image:
                    origin_features = (self.model.clip.encode_text(text).detach())
                    target_features= self.model.clip.encode_text(target_text).detach()

                    # origin_features, _, _, _ = self.model.clip.encode_text(text)
                    # origin_features = origin_features.detach()
                    # target_features, _, _, _ = self.model.clip.encode_text(target_text)
                    # target_features = target_features.detach()
                    if self.args.model == "DNPH":
                        origin_hash,_ = self.model.encode_text(text)
                        target_hash,_= self.model.encode_text(target_text)
                    #DSPH DCHMT
                    else:
                        origin_hash= self.model.encode_text(text)
                        target_hash= self.model.encode_text(target_text)
                        # target_hash = self.model.encode_image(target_image)

                if self.args.is_text:
                    origin_features = self.model.clip.encode_image(image).detach()
                    target_features = self.model.clip.encode_image(target_image).detach()
                    if self.args.model == "DNPH":
                        origin_hash, _ = self.model.encode_image(image)
                        target_hash, _ = self.model.encode_image(target_image)
                    else:
                        origin_hash = self.model.encode_image(image)
                        target_hash = self.model.encode_image(target_image)

                # Generate adversarial samples
                if self.args.is_image:
                    adv_image = self.image_attack(num ,image=image, target_text = target_text, query_labels=target_labels, index=index, target_features=target_features, origin_features = origin_features, target_hash = target_hash,
                        origin_hash = origin_hash, epsilon=self.args.epsilon, alpha=self.args.alpha, num_iter=self.args.num_iter,
                    )
                    num = num + 1
                    if self.args.model == "DNPH":
                        image_hash,_ = self.model.encode_image(adv_image)
                        text_hash,_ = self.model.encode_text(text)

                        # trans
                        # image_hash = self.model3.encode_image(adv_image)
                        # text_hash = self.model3.encode_text(text)

                    else:
                        image_hash = self.model.encode_image(adv_image)
                        # image_hash,_ = self.model3.encode_image(adv_image)

                        text_hash = self.model.encode_text(text)

                if self.args.is_text:
                    adv_text = self.text_attack(
                        text=text,
                        target_image=target_image, query_labels=target_labels, index=index,
                        target_features=target_features,
                        origin_features = origin_features,
                        target_hash = target_hash,
                        origin_hash = origin_hash,
                        epsilon=self.args.epsilon,
                        alpha=self.args.alpha,
                        num_iter=self.args.num_iter,
                    )
                    # # adv_text = self.tokenizer(adv_text,padding='max_length', truncation=True, max_length=30,  return_tensors='pt').to(0)
                    # image_hash = self.model.encode_image(image)
                    # text_hash = self.model.encode_text(adv_text)

                    if self.args.model == "DNPH":
                        image_hash,_ = self.model.encode_image(image)
                        text_hash,_ = self.model.encode_text(adv_text,1)

                    else:
                        image_hash = self.model.encode_image(image)
                        text_hash = self.model.encode_text(adv_text,1)


                        # text_hash,_ = self.model3.encode_text(adv_text,1)


                if self.args.model == "DCMHT":
                    img_buffer[target_index, :] = image_hash.data.sign_()
                    text_buffer[target_index, :] = text_hash.data.sign_()

                    #text
                    # text_buffer[target_index, :] = text_hash.data[:, -1, :]  # Shape becomes [32,64]

                else:
                    img_buffer[target_index, :] = image_hash.data
                    text_buffer[target_index, :] = text_hash.data

                    #text
                    # text_buffer[target_index, :] = text_hash.data[:, -1, :]   # Shape becomes [32,64]


        else:
            for image, text, label, index in tqdm(data_loader):
                image = image.to(self.rank, non_blocking=True)
                text = text.to(self.rank, non_blocking=True)
                index = index.numpy()

                if self.args.is_attack:
                    target_features = self.model.clip.encode_text(text).detach()
                    # Generate adversarial samples
                    adv_image = self.image_attack(image=image, target=target_features, epsilon=0.01, alpha=0.001, num_iter=50)

                    image_hash = self.model.encode_image(adv_image)
                    text_hash = self.model.encode_text(text)

                    if self.args.model == "DCMHT":
                        img_buffer[index, :] = image_hash.data.sign_()
                        text_buffer[index, :] = text_hash.data.sign_()
                    else:
                        img_buffer[index, :] = image_hash.data
                        text_buffer[index, :] = text_hash.data
                else:
                    if self.args.model == "DNPH":
                        image_hash,_ = self.model.encode_image(image)
                        text_hash,_ = self.model.encode_text(text)

                        # trans
                        # image_hash = self.model3.encode_image(image)
                        # text_hash = self.model3.encode_text(text)
                    else:
                        image_hash = self.model.encode_image(image)
                        text_hash = self.model.encode_text(text)

                        #trans
                        # image_hash,_ = self.model3.encode_image(image)
                        # text_hash,_ = self.model3.encode_text(text)

                    if self.args.model == "DCMHT":
                        img_buffer[index, :] = image_hash.data.sign_()
                        text_buffer[index, :] = text_hash.data.sign_()
                    else:
                        img_buffer[index, :] = image_hash.data
                        text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer  # img_buffer.to(self.rank), text_buffer.to(self.rank)

    def sensitivity_analysis(self):
        """
        Run sensitivity analysis for epsilon, alpha, and num_iter.
        Results are automatically saved as JSON.
        """

        epsilon_list = [1/255 , 2 / 255, 4 / 255, 8 / 255, 16 / 255]
        alpha_list = [0.002, 0.004, 0.008, 0.016, 0.032]
        iter_list = [20,40,60,80,100,200,300]

        # Default baseline parameters, called the default setting in the paper
        base_epsilon = self.args.epsilon
        base_alpha = self.args.alpha
        base_iter = self.args.num_iter

        results = {
            "epsilon_sensitivity": [],
            "alpha_sensitivity": [],
            "iter_sensitivity": []
        }

        # ========== epsilon ==========
        for eps in epsilon_list:
            self.logger.info(f"[Sensitivity] epsilon={eps}")
            self.args.epsilon = eps
            self.args.alpha = base_alpha
            self.args.num_iter = base_iter

            metrics = self.run_single_attack()
            results["epsilon_sensitivity"].append({
                "epsilon": eps,
                **metrics
            })

        # ========== alpha ==========
        for alpha in alpha_list:
            self.logger.info(f"[Sensitivity] alpha={alpha}")
            self.args.epsilon = base_epsilon
            self.args.alpha = alpha
            self.args.num_iter = base_iter

            metrics = self.run_single_attack()
            results["alpha_sensitivity"].append({
                "alpha": alpha,
                **metrics
            })

        # ========== num_iter ==========
        for it in iter_list:
            self.logger.info(f"[Sensitivity] num_iter={it}")
            self.args.epsilon = base_epsilon
            self.args.alpha = base_alpha
            self.args.num_iter = it

            metrics = self.run_single_attack()
            results["iter_sensitivity"].append({
                "num_iter": it,
                **metrics
            })

        self.save_sensitivity_results(results)

    def run_single_attack(self):
        """
        Run one complete attack and return the core evaluation metrics.
        """

        self.change_state(mode="valid")

        # Original query
        query_img, query_txt = self.get_attack_code(
            self.query_loader, None, self.args.query_num
        )

        # target
        target_img, target_txt = self.get_attack_code(
            self.target_loader, None, self.args.query_num
        )

        # Attacked query
        query_advimg, query_advtxt = self.get_attack_code(
            self.query_loader, self.target_loader, self.args.query_num
        )

        # retrieval
        retrieval_img, retrieval_txt = self.get_attack_code(
            self.retrieval_loader, None, self.args.retrieval_num
        )

        # ===== mAP =====
        adv_mAPi2t = calc_map_k(
            query_advimg, retrieval_txt,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )
        adv_mAPt2i = calc_map_k(
            query_advtxt, retrieval_img,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )
        adv_mAPi2i = calc_map_k(
            query_advimg, retrieval_img,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )
        adv_mAPt2t = calc_map_k(
            query_advtxt, retrieval_txt,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )

        return {
            "Adv_mAP_i2t": float(adv_mAPi2t),
            "Adv_mAP_t2i": float(adv_mAPt2i),
            "Adv_mAP_i2i": float(adv_mAPi2i),
            "Adv_mAP_t2t": float(adv_mAPt2t),
        }

    def save_sensitivity_results(self, results, filename=None):
        """
        Save sensitivity analysis results.
        """

        if filename is None:
            filename = f"sensitivity_{self.args.model}_{self.args.dataset}.json"

        save_path = os.path.join(self.args.save_dir, filename)

        save_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.args.model,
            "dataset": self.args.dataset,
            "base_setting": {
                "epsilon": self.args.epsilon,
                "alpha": self.args.alpha,
                "num_iter": self.args.num_iter
            },
            "results": results
        }

        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=4)

        self.logger.info(f"Sensitivity results saved to {save_path}")


    def sensitivity_analysis_token(self):
        """
        Run sensitivity analysis for epsilon, alpha, num_iter, and top_k.
        Results are automatically saved as JSON.
        """

        top_k_list = list(range(1, 9))  # token top_k

        # Baseline parameters
        base_epsilon = self.args.epsilon
        base_alpha = self.args.alpha
        base_iter = self.args.num_iter


        results = {
            "token_sensitivity": []
        }

        # ===== top_k =====
        for k in top_k_list:
            self.logger.info(f"[Sensitivity] token={k}")
            self.args.epsilon = base_epsilon
            self.args.alpha = base_alpha
            self.args.num_iter = base_iter
            self.args.l5 = k * 100

            # Call the modified run_single_attack_for_topk
            metrics = self.run_single_attack_token()
            results["token_sensitivity"].append({"token": self.args.l5, **metrics})

        self.save_sensitivity_results_token(results)

    def run_single_attack_token(self):
        """
        Run one complete attack and return the core evaluation metrics.
        """

        self.change_state(mode="valid")

        # Original query
        query_img, query_txt = self.get_attack_code(
            self.query_loader, None, self.args.query_num
        )

        # target
        target_img, target_txt = self.get_attack_code(
            self.target_loader, None, self.args.query_num
        )

        # Attacked query
        query_advimg, query_advtxt = self.get_attack_code(
            self.query_loader, self.target_loader, self.args.query_num
        )

        # retrieval
        retrieval_img, retrieval_txt = self.get_attack_code(
            self.retrieval_loader, None, self.args.retrieval_num
        )

        # ===== mAP =====
        adv_mAPi2t = calc_map_k(
            query_advimg, retrieval_txt,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )
        adv_mAPt2i = calc_map_k(
            query_advtxt, retrieval_img,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )
        adv_mAPi2i = calc_map_k(
            query_advimg, retrieval_img,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )
        adv_mAPt2t = calc_map_k(
            query_advtxt, retrieval_txt,
            self.target_labels, self.retrieval_labels,
            self.args.k, self.rank
        )

        return {
            "Adv_mAP_i2t": float(adv_mAPi2t),
            "Adv_mAP_t2i": float(adv_mAPt2i),
            "Adv_mAP_i2i": float(adv_mAPi2i),
            "Adv_mAP_t2t": float(adv_mAPt2t),
        }

    def save_sensitivity_results_token(self, results, filename=None):
        """
        Save sensitivity analysis results.
        """

        if filename is None:
            filename = f"sensitivity_l5_{self.args.model}_{self.args.dataset}.json"

        save_path = os.path.join(self.args.save_dir, filename)

        save_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.args.model,
            "dataset": self.args.dataset,
            "base_setting": {
                "epsilon": self.args.epsilon,
                "alpha": self.args.alpha,
                "num_iter": self.args.num_iter,
                "token": self.args.token,
                "l5": self.args.l5

            },
            "results": results
        }

        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=4)

        self.logger.info(f"Sensitivity results saved to {save_path}")



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

    # cfg.run.update({"save_dir": args.save_dir})

    return cfg
