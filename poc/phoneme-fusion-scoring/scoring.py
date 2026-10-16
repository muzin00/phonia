"""Same Transformer parameters; either cosine weighting or a direct residual score."""

from shared import ARMS, load_encoder, rt


class DirectFusion(rt.backend.Fusion):
    def forward(self, features, components, mask, phone_ids=None):
        if not mask.any(dim=1).all():
            raise ValueError("At least one shared phone is required")
        if phone_ids is None:
            phone_ids = rt.torch.arange(mask.shape[1], device=features.device)
        clean = rt.torch.where(mask[..., None], features, 0)
        state = self.input(clean) + self.phone(phone_ids)[None]
        state = self.context(state, src_key_padding_mask=~mask)
        residual = self.output(state).squeeze(-1)
        uniform = mask.to(features.dtype) / mask.sum(1, keepdim=True)
        base = (uniform * rt.torch.where(mask, components, 0)).sum(1)
        correction = (uniform * rt.torch.where(mask, residual, 0)).sum(1)
        # The second result preserves the legacy API. These uniform coefficients
        # are NOT an attribution of the contextual residual or attention weights.
        return base + correction, uniform

    def loss(self, features, components, mask, genuine, regularization=0.0):
        if regularization != 0:
            raise ValueError("Direct scoring has no phoneme-weight KL penalty")
        score, _ = self(features, components, mask)
        logits = self.log_scale.exp().clamp(1, 100) * score + self.bias
        return rt.torch.nn.functional.binary_cross_entropy_with_logits(logits, genuine)


class FrozenModel:
    def __init__(self, config, arm, seed):
        if arm not in ARMS:
            raise ValueError("Unknown scoring arm")
        self.config, self.arm, self.seed = config, arm, seed
        self.encoder = load_encoder(config, seed)
        self.reset_fusion()

    def reset_fusion(self):
        rt.torch.manual_seed(rt.seed_for(self.seed, "fusion-initialization"))
        cls = DirectFusion if self.arm == "direct" else rt.backend.Fusion
        self.fusion = cls("transformer")
        self.fusion_optimizer = rt.torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.config["fusion_learning_rate"],
            weight_decay=self.config["fusion_weight_decay"],
        )


def arm_configuration(config, arm):
    return {
        **config,
        "scoring_arm": arm,
        "fusion_uniform_kl": config["fusion_uniform_kl"] if arm == "weighted" else 0.0,
    }
