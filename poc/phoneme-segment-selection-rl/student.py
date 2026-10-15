"""Fresh acoustic encoder plus separately updated fusion, with complete forks."""

from __future__ import annotations

import math

from runtime import (
    AAMSoftmax,
    acoustic,
    backend,
    clone,
    create_encoder,
    np,
    seed_for,
    torch,
)


class Student:
    def __init__(self, config, speakers, seed):
        self.config, self.speakers, self.seed = config, speakers, seed
        torch.manual_seed(seed)
        self.encoder = create_encoder("statistics_mlp")
        self.head = AAMSoftmax(speakers)
        self.optimizer = torch.optim.AdamW(
            [*self.encoder.parameters(), *self.head.parameters()],
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.reset_fusion()
        self.update = 0

    def reset_fusion(self):
        torch.manual_seed(seed_for(self.seed, "fusion-initialization"))
        self.fusion = backend.Fusion("transformer")
        self.fusion_optimizer = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.config["fusion_learning_rate"],
            weight_decay=self.config["fusion_weight_decay"],
        )

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "fusion": self.fusion.state_dict(),
            "fusion_optimizer": self.fusion_optimizer.state_dict(),
            "update": self.update,
            "seed": self.seed,
            "speakers": self.speakers,
        }

    def load_state_dict(self, state):
        if (state["seed"], state["speakers"]) != (self.seed, self.speakers):
            raise ValueError("Student identity changed")
        for key in ("encoder", "head", "optimizer", "fusion", "fusion_optimizer"):
            getattr(self, key).load_state_dict(state[key])
        self.update = state["update"]

    def fork(self):
        child = Student(self.config, self.speakers, self.seed)
        child.load_state_dict(clone(self.state_dict()))
        return child

    def learning_rate(self, update):
        c = self.config
        if c["warmup_updates"] and update < c["warmup_updates"]:
            return c["learning_rate"] * (update + 1) / c["warmup_updates"]
        progress = min(
            1.0,
            max(
                0.0,
                (update - c["warmup_updates"]) / (c["updates"] - c["warmup_updates"]),
            ),
        )
        return c["minimum_learning_rate"] + 0.5 * (
            c["learning_rate"] - c["minimum_learning_rate"]
        ) * (1 + math.cos(math.pi * progress))

    def step(self, pool, indices, valid, update, singleton=None):
        torch.manual_seed(seed_for(self.seed, "acoustic-dropout", update))
        self.encoder.train()
        self.head.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate(update)
        values = torch.from_numpy(np.array(pool.features[np.maximum(indices, 0)]))
        labels = torch.from_numpy(
            pool.segment_speaker[np.maximum(indices, 0)].astype(np.int64)
        )
        mask = torch.from_numpy(valid)
        embedded = torch.nn.functional.normalize(
            self.encoder.projection(values), dim=1, eps=self.encoder.l2_epsilon
        )
        loss = acoustic.grouped_aam(
            self.head, embedded, labels, mask
        ) + 0.5 * acoustic.grouped_supcon(embedded, mask)
        if singleton is not None:
            x = torch.from_numpy(np.array(pool.features[singleton : singleton + 1]))
            z = torch.nn.functional.normalize(
                self.encoder.projection(x), dim=1, eps=self.encoder.l2_epsilon
            )
            # Singleton groups cannot create genuine positives. Identical auxiliary
            # classification schedule in every arm; never fake a contrastive pair.
            loss = loss + 0.01 * self.head(
                z, torch.tensor([int(pool.segment_speaker[singleton])])
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*self.encoder.parameters(), *self.head.parameters()],
            self.config["gradient_clip"],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.update = update + 1
        return float(loss.detach())
