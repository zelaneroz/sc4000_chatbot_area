# reward_modeling_experiments.py

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =========================================================
# 1) CONFIGS
# =========================================================

@dataclass
class HyperParams:
    learning_rate: float
    weight_decay: float
    dropout: float
    batch_size: int
    num_epochs: int
    max_length: int
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True


@dataclass
class SetupConfig:
    name: str
    model_name: str
    architecture_family: str  # "encoder" or "decoder"
    objective_type: str       # "pairwise" or "regression"
    loss_type: str            # "bradley_terry", "sigmoid_ce", "mse"
    features: List[str]       # e.g. ["prompt", "response", "length_features"]
    hyperparameters: HyperParams
    seed: int = 42
    use_wandb: bool = False
    project_name: str = "reward-modeling"
    output_dir: str = "./outputs"
    target_mode: str = "three_way"  # "three_way" or "binary_a_vs_b"
    peft_mode: Optional[str] = None # placeholder for LoRA/QLoRA if needed


def build_experiment_matrix() -> Dict[str, SetupConfig]:
    return {
        "setup_1": SetupConfig(
            name="setup_1_deberta_pairwise_bt",
            model_name="microsoft/deberta-v3-base",
            architecture_family="encoder",
            objective_type="pairwise",
            loss_type="bradley_terry",
            features=["prompt", "response"],
            hyperparameters=HyperParams(
                learning_rate=2e-5,
                weight_decay=0.01,
                dropout=0.10,
                batch_size=8,
                num_epochs=2,
                max_length=512,
            ),
        ),
        "setup_2": SetupConfig(
            name="setup_2_deberta_pairwise_bt_len",
            model_name="microsoft/deberta-v3-base",
            architecture_family="encoder",
            objective_type="pairwise",
            loss_type="bradley_terry",
            features=["prompt", "response", "length_features"],
            hyperparameters=HyperParams(
                learning_rate=2e-5,
                weight_decay=0.01,
                dropout=0.10,
                batch_size=8,
                num_epochs=2,
                max_length=512,
            ),
        ),
        "setup_3": SetupConfig(
            name="setup_3_deberta_pairwise_sigmoid",
            model_name="microsoft/deberta-v3-base",
            architecture_family="encoder",
            objective_type="pairwise",
            loss_type="sigmoid_ce",
            features=["prompt", "response"],
            hyperparameters=HyperParams(
                learning_rate=1.5e-5,
                weight_decay=0.02,
                dropout=0.15,
                batch_size=8,
                num_epochs=2,
                max_length=512,
            ),
        ),
        "setup_4": SetupConfig(
            name="setup_4_deberta_regression_mse",
            model_name="microsoft/deberta-v3-base",
            architecture_family="encoder",
            objective_type="regression",
            loss_type="mse",
            features=["prompt", "response", "length_features"],
            hyperparameters=HyperParams(
                learning_rate=1e-5,
                weight_decay=0.01,
                dropout=0.10,
                batch_size=8,
                num_epochs=2,
                max_length=512,
            ),
        ),
        "setup_5": SetupConfig(
            name="setup_5_llama_pairwise_bt",
            model_name="meta-llama/Llama-2-7b-hf",  # replace with your accessible decoder model
            architecture_family="decoder",
            objective_type="pairwise",
            loss_type="bradley_terry",
            features=["prompt", "response"],
            hyperparameters=HyperParams(
                learning_rate=2e-5,
                weight_decay=0.01,
                dropout=0.05,
                batch_size=2,
                num_epochs=1,
                max_length=1024,
            ),
            peft_mode="lora",
        ),
        "setup_6": SetupConfig(
            name="setup_6_llama_regression_mse_len",
            model_name="meta-llama/Llama-2-7b-hf",
            architecture_family="decoder",
            objective_type="regression",
            loss_type="mse",
            features=["prompt", "response", "length_features"],
            hyperparameters=HyperParams(
                learning_rate=1e-5,
                weight_decay=0.00,
                dropout=0.05,
                batch_size=2,
                num_epochs=1,
                max_length=1024,
            ),
            peft_mode="lora",
        ),
    }


# =========================================================
# 2) UTILS
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def winner_to_class(row: pd.Series) -> int:
    # 0=A, 1=B, 2=tie
    if row["winner_model_a"] == 1:
        return 0
    if row["winner_model_b"] == 1:
        return 1
    if row["winner_tie"] == 1:
        return 2
    raise ValueError("Invalid target row")


def pairwise_target_value(row: pd.Series) -> float:
    # for pairwise ranking:
    # A wins -> 1.0
    # B wins -> 0.0
    # tie -> 0.5
    if row["winner_model_a"] == 1:
        return 1.0
    if row["winner_model_b"] == 1:
        return 0.0
    if row["winner_tie"] == 1:
        return 0.5
    raise ValueError("Invalid target row")


def regression_reward_targets(row: pd.Series) -> Tuple[float, float]:
    # scalar reward targets for A and B
    # A wins -> (1,0), B wins -> (0,1), tie -> (0.5,0.5)
    if row["winner_model_a"] == 1:
        return 1.0, 0.0
    if row["winner_model_b"] == 1:
        return 0.0, 1.0
    if row["winner_tie"] == 1:
        return 0.5, 0.5
    raise ValueError("Invalid target row")


def estimate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


# =========================================================
# 3) DATASET
# =========================================================

class PreferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        config: SetupConfig,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.hyperparameters.max_length
        self.use_length_features = "length_features" in config.features

    def __len__(self) -> int:
        return len(self.df)

    def _format_text(self, prompt: str, response: str) -> str:
        if "prompt" in self.config.features and "response" in self.config.features:
            return f"Prompt:\n{prompt}\n\nResponse:\n{response}"
        if "prompt" in self.config.features and "response" not in self.config.features:
            return f"Prompt:\n{prompt}"
        if "response" in self.config.features and "prompt" not in self.config.features:
            return f"Response:\n{response}"
        return f"Prompt:\n{prompt}\n\nResponse:\n{response}"

    def _length_feats(self, row: pd.Series) -> np.ndarray:
        a_chars = len(row["response_a_text"])
        b_chars = len(row["response_b_text"])
        a_words = len(row["response_a_text"].split())
        b_words = len(row["response_b_text"].split())
        a_tok = estimate_token_count(row["response_a_text"])
        b_tok = estimate_token_count(row["response_b_text"])
        feats = np.array([
            a_chars, b_chars, a_chars - b_chars,
            a_words, b_words, a_words - b_words,
            a_tok, b_tok, a_tok - b_tok,
        ], dtype=np.float32)
        return feats

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        prompt = row["prompt_text"]
        response_a = row["response_a_text"]
        response_b = row["response_b_text"]

        text_a = self._format_text(prompt, response_a)
        text_b = self._format_text(prompt, response_b)

        enc_a = self.tokenizer(
            text_a,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc_b = self.tokenizer(
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        example = {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "pairwise_target": pairwise_target_value(row),
            "winner_class": winner_to_class(row),
            "id": row.get("example_id", row["id"]),
        }

        if self.config.objective_type == "regression":
            ra, rb = regression_reward_targets(row)
            example["reward_a"] = ra
            example["reward_b"] = rb

        if self.use_length_features:
            example["length_features"] = self._length_feats(row)

        return example


class PreferenceCollator:
    def __init__(self, tokenizer, use_length_features: bool = False):
        self.tokenizer = tokenizer
        self.use_length_features = use_length_features

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_a = [{"input_ids": x["input_ids_a"], "attention_mask": x["attention_mask_a"]} for x in batch]
        batch_b = [{"input_ids": x["input_ids_b"], "attention_mask": x["attention_mask_b"]} for x in batch]

        pad_a = self.tokenizer.pad(batch_a, padding=True, return_tensors="pt")
        pad_b = self.tokenizer.pad(batch_b, padding=True, return_tensors="pt")

        out = {
            "input_ids_a": pad_a["input_ids"],
            "attention_mask_a": pad_a["attention_mask"],
            "input_ids_b": pad_b["input_ids"],
            "attention_mask_b": pad_b["attention_mask"],
            "pairwise_target": torch.tensor([x["pairwise_target"] for x in batch], dtype=torch.float32),
            "winner_class": torch.tensor([x["winner_class"] for x in batch], dtype=torch.long),
            "id": [x["id"] for x in batch],
        }

        if "reward_a" in batch[0]:
            out["reward_a"] = torch.tensor([x["reward_a"] for x in batch], dtype=torch.float32)
            out["reward_b"] = torch.tensor([x["reward_b"] for x in batch], dtype=torch.float32)

        if self.use_length_features:
            out["length_features"] = torch.tensor(
                np.stack([x["length_features"] for x in batch]),
                dtype=torch.float32,
            )

        return out


# =========================================================
# 4) MODEL
# =========================================================

class RewardBackbone(nn.Module):
    """
    Shared wrapper around encoder/decoder backbones.
    Produces one scalar reward per (prompt, response) input.
    """

    def __init__(self, config: SetupConfig, length_feature_dim: int = 9):
        super().__init__()
        self.setup = config
        hf_cfg = AutoConfig.from_pretrained(config.model_name)
        hf_cfg.hidden_dropout_prob = config.hyperparameters.dropout if hasattr(hf_cfg, "hidden_dropout_prob") else getattr(hf_cfg, "hidden_dropout_prob", 0.1)
        hf_cfg.attention_probs_dropout_prob = config.hyperparameters.dropout if hasattr(hf_cfg, "attention_probs_dropout_prob") else getattr(hf_cfg, "attention_probs_dropout_prob", 0.1)

        self.backbone = AutoModel.from_pretrained(config.model_name, config=hf_cfg)
        hidden_size = getattr(hf_cfg, "hidden_size", None) or getattr(hf_cfg, "dim", None) or getattr(hf_cfg, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone config.")

        self.use_length_features = "length_features" in config.features
        head_in = hidden_size + (length_feature_dim if self.use_length_features else 0)

        self.dropout = nn.Dropout(config.hyperparameters.dropout)
        self.reward_head = nn.Sequential(
            nn.Linear(head_in, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(config.hyperparameters.dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def _pool(self, outputs, attention_mask):
        # General-purpose masked mean pooling
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward_once(self, input_ids, attention_mask, length_features=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs, attention_mask)
        pooled = self.dropout(pooled)

        if self.use_length_features and length_features is not None:
            pooled = torch.cat([pooled, length_features], dim=-1)

        reward = self.reward_head(pooled).squeeze(-1)
        return reward

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        length_features=None,
    ):
        reward_a = self.forward_once(input_ids_a, attention_mask_a, length_features)
        reward_b = self.forward_once(input_ids_b, attention_mask_b, length_features)
        return reward_a, reward_b


# =========================================================
# 5) LOSSES
# =========================================================

def bradley_terry_loss(reward_a, reward_b, target):
    # target in {1.0, 0.5, 0.0}
    # p(a preferred) = sigmoid(r_a - r_b)
    diff = reward_a - reward_b
    prob_a = torch.sigmoid(diff)

    # binary CE for hard outcomes, midpoint penalty for ties
    # tie target = 0.5
    loss = nn.functional.binary_cross_entropy(prob_a, target)
    return loss, prob_a


def sigmoid_cross_entropy_loss(reward_a, reward_b, target):
    # mathematically similar, but kept separate for explicit setup routing
    diff = reward_a - reward_b
    prob_a = torch.sigmoid(diff)
    loss = nn.functional.binary_cross_entropy(prob_a, target)
    return loss, prob_a


def mse_reward_loss(reward_a, reward_b, target_a, target_b):
    loss_a = nn.functional.mse_loss(reward_a, target_a)
    loss_b = nn.functional.mse_loss(reward_b, target_b)
    loss = 0.5 * (loss_a + loss_b)
    prob_a = torch.sigmoid(reward_a - reward_b)
    return loss, prob_a


# =========================================================
# 6) METRICS
# =========================================================

def probs_from_pairwise(prob_a: np.ndarray, tie_calibration: bool = True) -> np.ndarray:
    """
    Convert scalar P(A>B) into 3-way probs [A, B, tie].
    Tie handling is heuristic boilerplate:
    more uncertainty near 0.5 -> higher tie probability.
    """
    if tie_calibration:
        tie_prob = 1.0 - np.abs(prob_a - 0.5) * 2.0
        tie_prob = np.clip(tie_prob * 0.20, 0.0, 0.20)  # cap tie mass for baseline
    else:
        tie_prob = np.zeros_like(prob_a)

    remaining = 1.0 - tie_prob
    p_a = remaining * prob_a
    p_b = remaining * (1.0 - prob_a)

    probs = np.stack([p_a, p_b, tie_prob], axis=1)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def compute_validation_metrics(y_true_class: np.ndarray, prob_a: np.ndarray) -> Dict[str, float]:
    probs_3 = probs_from_pairwise(prob_a)
    y_pred_class = probs_3.argmax(axis=1)

    acc = accuracy_score(y_true_class, y_pred_class)
    ll = log_loss(y_true_class, probs_3, labels=[0, 1, 2])

    return {
        "val_accuracy": float(acc),
        "val_log_loss": float(ll),
    }


# =========================================================
# 7) TRACKER
# =========================================================

class ExperimentTracker:
    def __init__(
        self,
        output_dir: str,
        project_name: str,
        use_wandb: bool = False,
    ):
        self.output_dir = output_dir
        self.project_name = project_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        ensure_dir(output_dir)
        self.summary_csv = os.path.join(output_dir, "experiment_summary.csv")
        self.pred_dir = os.path.join(output_dir, "predictions")
        ensure_dir(self.pred_dir)

        if not os.path.exists(self.summary_csv):
            with open(self.summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "setup_name",
                    "model_name",
                    "objective_type",
                    "loss_type",
                    "val_accuracy",
                    "val_log_loss",
                    "elo_rating",
                ])

    def start_run(self, cfg: SetupConfig):
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                name=cfg.name,
                config=asdict(cfg),
                reinit=True,
            )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def finish_run(self):
        if self.use_wandb:
            wandb.finish()

    def save_predictions(self, setup_name: str, ids: List[Any], y_true: np.ndarray, prob_a: np.ndarray):
        df = pd.DataFrame({
            "id": ids,
            "y_true_class": y_true,
            "prob_a_pref": prob_a,
        })
        path = os.path.join(self.pred_dir, f"{setup_name}_val_predictions.csv")
        df.to_csv(path, index=False)

    def append_summary(self, row: Dict[str, Any]):
        with open(self.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row["setup_name"],
                row["model_name"],
                row["objective_type"],
                row["loss_type"],
                row["val_accuracy"],
                row["val_log_loss"],
                row.get("elo_rating", np.nan),
            ])


def compute_elo_from_prediction_files(pred_dir: str, k_factor: float = 32.0) -> Dict[str, float]:
    """
    Elo over setups, based on per-example negative log-likelihood.
    For each pair of setups, on each shared example:
    - lower NLL gets a 'win'
    - equal gets a draw
    """
    files = sorted(Path(pred_dir).glob("*_val_predictions.csv"))
    if len(files) < 2:
        return {}

    preds = {}
    for fp in files:
        name = fp.stem.replace("_val_predictions", "")
        df = pd.read_csv(fp)
        probs3 = probs_from_pairwise(df["prob_a_pref"].values)
        # per-example NLL
        y = df["y_true_class"].values.astype(int)
        nll = -np.log(np.clip(probs3[np.arange(len(y)), y], 1e-15, 1.0))
        preds[name] = pd.DataFrame({"id": df["id"], "nll": nll})

    ratings = {name: 1500.0 for name in preds.keys()}

    names = list(preds.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            merged = preds[a].merge(preds[b], on="id", suffixes=("_a", "_b"))
            for _, row in merged.iterrows():
                ra, rb = ratings[a], ratings[b]
                ea = 1 / (1 + 10 ** ((rb - ra) / 400))
                eb = 1 - ea

                if row["nll_a"] < row["nll_b"]:
                    sa, sb = 1.0, 0.0
                elif row["nll_a"] > row["nll_b"]:
                    sa, sb = 0.0, 1.0
                else:
                    sa, sb = 0.5, 0.5

                ratings[a] = ra + k_factor * (sa - ea)
                ratings[b] = rb + k_factor * (sb - eb)

    return ratings


# =========================================================
# 8) TRAIN / EVAL
# =========================================================

def build_optimizer_and_scheduler(model, cfg: SetupConfig, total_steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    )
    warmup_steps = int(total_steps * cfg.hyperparameters.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, device, cfg, tracker=None, epoch=0):
    model.train()
    total_loss = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.hyperparameters.mixed_precision and torch.cuda.is_available()))

    for step, batch in enumerate(loader):
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)

        length_features = batch.get("length_features")
        if length_features is not None:
            length_features = length_features.to(device)

        with torch.cuda.amp.autocast(enabled=(cfg.hyperparameters.mixed_precision and torch.cuda.is_available())):
            reward_a, reward_b = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                length_features=length_features,
            )

            if cfg.objective_type == "pairwise":
                target = batch["pairwise_target"].to(device)
                if cfg.loss_type == "bradley_terry":
                    loss, _ = bradley_terry_loss(reward_a, reward_b, target)
                elif cfg.loss_type == "sigmoid_ce":
                    loss, _ = sigmoid_cross_entropy_loss(reward_a, reward_b, target)
                else:
                    raise ValueError(f"Unsupported pairwise loss: {cfg.loss_type}")

            elif cfg.objective_type == "regression":
                target_a = batch["reward_a"].to(device)
                target_b = batch["reward_b"].to(device)
                if cfg.loss_type == "mse":
                    loss, _ = mse_reward_loss(reward_a, reward_b, target_a, target_b)
                else:
                    raise ValueError(f"Unsupported regression loss: {cfg.loss_type}")

            else:
                raise ValueError(f"Unsupported objective_type: {cfg.objective_type}")

            loss = loss / cfg.hyperparameters.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.hyperparameters.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()

        if tracker is not None:
            tracker.log_metrics({"train_loss": loss.item()}, step=epoch * len(loader) + step)

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()

    all_ids = []
    all_targets = []
    all_prob_a = []
    total_loss = 0.0

    for batch in loader:
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)

        length_features = batch.get("length_features")
        if length_features is not None:
            length_features = length_features.to(device)

        reward_a, reward_b = model(
            input_ids_a=input_ids_a,
            attention_mask_a=attention_mask_a,
            input_ids_b=input_ids_b,
            attention_mask_b=attention_mask_b,
            length_features=length_features,
        )

        if cfg.objective_type == "pairwise":
            target = batch["pairwise_target"].to(device)
            if cfg.loss_type == "bradley_terry":
                loss, prob_a = bradley_terry_loss(reward_a, reward_b, target)
            elif cfg.loss_type == "sigmoid_ce":
                loss, prob_a = sigmoid_cross_entropy_loss(reward_a, reward_b, target)
            else:
                raise ValueError(f"Unsupported pairwise loss: {cfg.loss_type}")

        elif cfg.objective_type == "regression":
            target_a = batch["reward_a"].to(device)
            target_b = batch["reward_b"].to(device)
            if cfg.loss_type == "mse":
                loss, prob_a = mse_reward_loss(reward_a, reward_b, target_a, target_b)
            else:
                raise ValueError(f"Unsupported regression loss: {cfg.loss_type}")

        total_loss += loss.item()

        all_ids.extend(batch["id"])
        all_targets.extend(batch["winner_class"].cpu().numpy().tolist())
        all_prob_a.extend(prob_a.detach().cpu().numpy().tolist())

    y_true = np.array(all_targets)
    prob_a = np.array(all_prob_a)

    metrics = compute_validation_metrics(y_true, prob_a)
    metrics["val_objective_loss"] = float(total_loss / max(len(loader), 1))

    return metrics, all_ids, y_true, prob_a


def train_reward_model(
    config: SetupConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Any]:
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = PreferenceDataset(train_df, tokenizer, config)
    val_ds = PreferenceDataset(val_df, tokenizer, config)
    collator = PreferenceCollator(tokenizer, use_length_features=("length_features" in config.features))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.hyperparameters.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.hyperparameters.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = RewardBackbone(config).to(device)

    total_steps = math.ceil(len(train_loader) * config.hyperparameters.num_epochs / config.hyperparameters.gradient_accumulation_steps)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)

    tracker = ExperimentTracker(
        output_dir=config.output_dir,
        project_name=config.project_name,
        use_wandb=config.use_wandb,
    )
    tracker.start_run(config)

    best_metrics = None
    best_state = None

    for epoch in range(config.hyperparameters.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config, tracker, epoch)
        val_metrics, ids, y_true, prob_a = evaluate(model, val_loader, device, config)

        epoch_metrics = {
            "epoch": epoch,
            "train_epoch_loss": train_loss,
            **val_metrics,
        }
        tracker.log_metrics(epoch_metrics, step=(epoch + 1) * len(train_loader))

        if best_metrics is None or val_metrics["val_log_loss"] < best_metrics["val_log_loss"]:
            best_metrics = val_metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            tracker.save_predictions(config.name, ids, y_true, prob_a)

    ckpt_path = os.path.join(config.output_dir, f"{config.name}_best.pt")
    if best_state is not None:
        torch.save(best_state, ckpt_path)

    tracker.finish_run()

    return {
        "setup_name": config.name,
        "model_name": config.model_name,
        "objective_type": config.objective_type,
        "loss_type": config.loss_type,
        **best_metrics,
        "checkpoint_path": ckpt_path,
    }


# =========================================================
# 9) EXPERIMENT ORCHESTRATION
# =========================================================

def run_experiment_matrix(
    processed_df: pd.DataFrame,
    setups: Dict[str, SetupConfig],
    fold: int = 0,
) -> pd.DataFrame:
    """
    Assumes processed_df already has:
    - prompt_text
    - response_a_text
    - response_b_text
    - fold
    - example_id or id
    - winner_model_a / winner_model_b / winner_tie
    """
    results = []

    for _, cfg in setups.items():
        print(f"\n==== Running {cfg.name} ====")
        train_df = processed_df[processed_df["fold"] != fold].copy()
        val_df = processed_df[processed_df["fold"] == fold].copy()

        summary = train_reward_model(cfg, train_df, val_df)
        results.append(summary)

    # Compute Elo across all saved validation predictions
    pred_dir = next(iter(setups.values())).output_dir + "/predictions"
    elo_ratings = compute_elo_from_prediction_files(pred_dir)

    final_rows = []
    tracker = ExperimentTracker(
        output_dir=next(iter(setups.values())).output_dir,
        project_name=next(iter(setups.values())).project_name,
        use_wandb=False,
    )

    for row in results:
        row["elo_rating"] = float(elo_ratings.get(row["setup_name"], 1500.0))
        tracker.append_summary(row)
        final_rows.append(row)

    result_df = pd.DataFrame(final_rows).sort_values(
        ["val_log_loss", "elo_rating"],
        ascending=[True, False],
    ).reset_index(drop=True)

    result_df.to_csv(
        os.path.join(next(iter(setups.values())).output_dir, "final_comparison.csv"),
        index=False,
    )
    return result_df


# =========================================================
# 10) EXAMPLE ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    # Example:
    # processed_df = pd.read_csv("./processed_preference_df.csv")
    # Must include the fold column and cleaned text columns from preprocessing.
    #
    # Minimal expected columns:
    # ['id', 'prompt_text', 'response_a_text', 'response_b_text',
    #  'winner_model_a', 'winner_model_b', 'winner_tie', 'fold']

    processed_df = pd.read_csv("./processed_preference_df.csv")
    setups = build_experiment_matrix()

    # global defaults for output / tracking
    for _, cfg in setups.items():
        cfg.output_dir = "./reward_model_runs"
        cfg.use_wandb = False  # set True if wandb is installed and configured
        cfg.project_name = "arena-preference-reward-modeling"

    final_comparison = run_experiment_matrix(
        processed_df=processed_df,
        setups=setups,
        fold=0,
    )

    print("\nFinal comparison:")
    print(final_comparison)
