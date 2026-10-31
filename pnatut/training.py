"""Masked-marker training: masking, loss on masked nodes only, the loop, and simple predictors.

Self-supervised objective: hide the marker of a random subset of molecules (replace their
one-hot row by the mask indicator), let the encoder see the rest of the patch, and
predict the hidden markers. The loss is cross-entropy **only on masked nodes**.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from pnatut.encoder import MaskedMarkerSAGE, apply_mask
from pnatut.pyg_compat import Batch, Data


@dataclass
class TrainConfig:
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    mask_rate: float = 0.15
    seed: int = 0
    class_weighting: str = "none"       # "none" | "inverse_sqrt"  (how to counter abundant markers)
    eval_mask_seed: int = 12345         # masks used for validation/test are fixed across models and controls
    device: str = "cpu"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_mask(batch: Tensor, rate: float, gen: torch.Generator) -> Tensor:
    """Boolean mask over nodes: each node masked with prob ``rate``; at least one node per graph."""
    n = batch.numel()
    mask = torch.rand(n, generator=gen) < rate
    n_graphs = int(batch.max()) + 1
    masked_per_graph = torch.zeros(n_graphs, dtype=torch.long).index_add_(0, batch, mask.long())
    for g in torch.nonzero(masked_per_graph == 0).view(-1).tolist():  # force one masked node in empty graphs
        nodes = torch.nonzero(batch == g).view(-1)
        mask[nodes[torch.randint(len(nodes), (1,), generator=gen)]] = True
    return mask


def masked_cross_entropy(logits: Tensor, y: Tensor, mask: Tensor, class_weight: Optional[Tensor] = None) -> Tensor:
    return nn.functional.cross_entropy(logits[mask], y[mask], weight=class_weight)


def class_weights(train_counts: np.ndarray, mode: str, max_ratio: float = 20.0) -> Optional[Tensor]:
    """``inverse_sqrt``: w_c = 1/sqrt(freq_c), capped at ``max_ratio`` times the smallest weight and then
    normalised to mean 1 (uncapped weights for markers seen a handful of times are in the hundreds and
    destabilise training); ``none``: no weighting."""
    if mode == "none":
        return None
    freq = train_counts / train_counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    w = np.minimum(w, w.min() * max_ratio)
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32)


def iterate_batches(datas: Sequence[Data], batch_size: int, shuffle: bool, gen: Optional[torch.Generator] = None):
    idx = torch.randperm(len(datas), generator=gen).tolist() if shuffle else list(range(len(datas)))
    for s in range(0, len(idx), batch_size):
        yield Batch.from_data_list([datas[i] for i in idx[s:s + batch_size]])


def fixed_masks(datas: Sequence[Data], rate: float, seed: int) -> List[Tensor]:
    """One deterministic mask per patch (so every model / control is scored on the same hidden nodes)."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for d in datas:
        m = torch.rand(d.num_nodes, generator=gen) < rate
        if not m.any():
            m[torch.randint(d.num_nodes, (1,), generator=gen)] = True
        out.append(m)
    return out


@torch.no_grad()
def evaluate_masked(model: MaskedMarkerSAGE, datas: Sequence[Data], masks: Sequence[Tensor], batch_size: int = 128,
                    class_weight: Optional[Tensor] = None) -> Dict[str, object]:
    """Loss, accuracy, balanced accuracy and per-class recall on the fixed masked nodes."""
    model.eval()
    V = model.n_markers
    tot_loss, n_masked = 0.0, 0
    correct = np.zeros(V, dtype=np.int64)
    support = np.zeros(V, dtype=np.int64)
    pred_counts = np.zeros(V, dtype=np.int64)
    for s in range(0, len(datas), batch_size):
        chunk = datas[s:s + batch_size]
        b = Batch.from_data_list(chunk)
        m = torch.cat(list(masks[s:s + batch_size]))
        xm = apply_mask(b.x, m, model.mask_column)
        out = model(xm, b.edge_index)
        loss = masked_cross_entropy(out["logits"], b.y, m, class_weight)
        k = int(m.sum())
        tot_loss += float(loss) * k
        n_masked += k
        pred = out["logits"][m].argmax(1)
        y = b.y[m]
        np.add.at(support, y.numpy(), 1)
        np.add.at(pred_counts, pred.numpy(), 1)
        np.add.at(correct, y[pred == y].numpy(), 1)
    present = support > 0
    recall = np.divide(correct, support, out=np.zeros(V), where=present)
    return {"loss": tot_loss / n_masked, "accuracy": correct.sum() / n_masked,
            "balanced_accuracy": recall[present].mean(), "n_masked": n_masked,
            "per_class": pd.DataFrame({"support": support, "correct": correct, "recall": recall, "n_predicted": pred_counts})}


def train_model(model: MaskedMarkerSAGE, train: Sequence[Data], val: Sequence[Data], cfg: TrainConfig,
                train_counts: Optional[np.ndarray] = None, verbose: bool = True) -> pd.DataFrame:
    """AdamW on the masked cross-entropy; keeps the weights of the best validation-loss epoch."""
    set_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    cw = class_weights(train_counts, cfg.class_weighting) if train_counts is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    val_masks = fixed_masks(val, cfg.mask_rate, cfg.eval_mask_seed)
    hist, best, best_state = [], np.inf, None
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for b in iterate_batches(train, cfg.batch_size, shuffle=True, gen=gen):
            m = choose_mask(b.batch, cfg.mask_rate, gen)
            xm = apply_mask(b.x, m, model.mask_column)
            out = model(xm, b.edge_index)
            loss = masked_cross_entropy(out["logits"], b.y, m, cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            k = int(m.sum())
            tot += loss.item() * k
            n += k
        ev = evaluate_masked(model, val, val_masks, class_weight=cw)
        row = {"epoch": ep, "train_loss": tot / n, "val_loss": ev["loss"], "val_accuracy": ev["accuracy"],
               "val_balanced_accuracy": ev["balanced_accuracy"], "seconds": time.time() - t0}
        hist.append(row)
        if ev["loss"] < best:
            best, best_state = ev["loss"], copy.deepcopy(model.state_dict())
        if verbose and (ep == 1 or ep % 5 == 0 or ep == cfg.epochs):
            print(f"epoch {ep:3d}  train_loss {row['train_loss']:.3f}  val_loss {row['val_loss']:.3f}  "
                  f"val_acc {row['val_accuracy']:.3f}  val_bal_acc {row['val_balanced_accuracy']:.3f}  ({row['seconds']:.1f}s)")
    model.load_state_dict(best_state)
    h = pd.DataFrame(hist).set_index("epoch")
    h.attrs["best_epoch"] = int(h["val_loss"].idxmin())
    return h


# ---- simple predictors for the same task --------------------------------------------------
def abundance_only_eval(train_counts: np.ndarray, datas: Sequence[Data], masks: Sequence[Tensor]) -> Dict[str, float]:
    """Predict every hidden marker with the training-set marker frequencies (ignores the patch entirely)."""
    p = np.maximum(train_counts / train_counts.sum(), 1e-9)
    y = torch.cat([d.y[m] for d, m in zip(datas, masks)]).numpy()
    return {"loss": float(-np.log(p[y]).mean()), "accuracy": float((y == p.argmax()).mean()),
            "balanced_accuracy": float(1.0 / (train_counts > 0).sum()), "predicts": int(p.argmax())}


class CompositionPredictor:
    """Composition-only predictor for the masked task: multinomial logistic regression from the *visible*
    marker proportions of a patch to each hidden marker. Sees counts only, never edges. Fit once on
    training patches, evaluate on anything (real or perturbed test patches)."""

    def __init__(self, n_markers: int, seed: int = 0, C: float = 1.0, max_iter: int = 300, use_pool: bool = False,
                 model: str = "logistic", hidden: int = 64):
        """``model``: "logistic" (linear, multinomial) or "mlp" (one hidden layer of ``hidden`` ReLU units,
        i.e. a non-linear function of the same edge-blind features, to separate model flexibility from topology)."""
        self.n_markers, self.seed, self.C, self.max_iter, self.use_pool = n_markers, seed, C, max_iter, use_pool
        self.model, self.hidden = model, hidden
        self.clf = None

    def _features(self, datas, masks):
        """Per masked molecule: visible marker proportions of its patch (+ its UMI pool as 2 indicator
        features when ``use_pool``; the pool is a property of the molecule, not of the edges)."""
        X, Y = [], []
        for d, m in zip(datas, masks):
            vis = np.bincount(d.y[~m].numpy(), minlength=self.n_markers).astype(np.float64)
            vis = vis / max(vis.sum(), 1)
            pools = d.pool[m].numpy() if self.use_pool else None
            for i, yy in enumerate(d.y[m].numpy()):
                row = vis if not self.use_pool else np.concatenate([vis, [pools[i] == 0, pools[i] == 1]])
                X.append(row)
                Y.append(yy)
        return np.asarray(X, dtype=np.float64), np.asarray(Y)

    def fit(self, train: Sequence[Data], train_masks: Sequence[Tensor]) -> "CompositionPredictor":
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        Xtr, Ytr = self._features(train, train_masks)
        if self.model == "logistic":
            self.clf = LogisticRegression(max_iter=self.max_iter, C=self.C, random_state=self.seed).fit(Xtr, Ytr)
        elif self.model == "mlp":
            self.clf = MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=self.max_iter, early_stopping=True,
                                     random_state=self.seed).fit(Xtr, Ytr)
        else:
            raise ValueError(self.model)
        self.n_train_rows = len(Ytr)
        return self

    def evaluate(self, datas: Sequence[Data], masks: Sequence[Tensor]) -> Dict[str, object]:
        Xev, Yev = self._features(datas, masks)
        P = np.full((len(Yev), self.n_markers), 1e-9)
        P[:, self.clf.classes_] = self.clf.predict_proba(Xev)
        P = P / P.sum(1, keepdims=True)
        pred = P.argmax(1)
        rec = [(pred[Yev == c] == c).mean() for c in np.unique(Yev)]
        return {"loss": float(-np.log(P[np.arange(len(Yev)), Yev]).mean()), "accuracy": float((pred == Yev).mean()),
                "balanced_accuracy": float(np.mean(rec)), "n_masked": int(len(Yev))}


def visible_composition_eval(train: Sequence[Data], train_masks: Sequence[Tensor], datas: Sequence[Data],
                             masks: Sequence[Tensor], n_markers: int, seed: int = 0) -> Dict[str, object]:
    """Convenience wrapper: fit :class:`CompositionPredictor` on ``train`` and evaluate on ``datas``."""
    cp = CompositionPredictor(n_markers, seed).fit(train, train_masks)
    out = cp.evaluate(datas, masks)
    out["n_train_rows"] = cp.n_train_rows
    return out
