"""CLI entry point modeled after the original Expert_Sparsity repository."""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
from argparse import Namespace
from datetime import datetime
from typing import Optional

import torch

from .data import DATASETS, build_calib_loader
from .method import METHODS

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(METHODS.keys()))
    parser.add_argument("--r", type=int, default=None, help="Number of experts to preserve")
    parser.add_argument("--calib_set", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--max_block_size", type=int, default=2048)
    parser.add_argument("--n_blocks_for_stat", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--score_metric", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--use_flash_attention_2", action="store_true")
    return parser


def parse_args() -> Namespace:
    return build_parser().parse_args()


def main(args: Optional[Namespace] = None):
    args = args or parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("Arguments: %s", args)

    model_name = args.model_path.rstrip("/").split("/")[-1]
    save_path = osp.join(
        args.output_path,
        f"{model_name}_{args.method}_{args.calib_set}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    os.makedirs(save_path, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None,
    )
    calib_loader = build_calib_loader(
        args.calib_set,
        tokenizer,
        args.max_block_size,
        args.n_blocks_for_stat,
        args.batch_size,
        args.num_workers,
        args.seed,
        dataset_path=args.dataset_path,
    )
    model, info = METHODS[args.method](model, calib_loader, args)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save((args, info), osp.join(save_path, "pruning_info.pt"))
    return model, info


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
