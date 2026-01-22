#!/usr/bin/env python3
"""
Multi-Branch Pipeline - Clean Architecture with Auto-Discovery

Usage:
    python entry/test_multi_branch_video.py
    python entry/test_multi_branch_video.py --config configs/multi-branch.yaml
"""

import argparse
import faulthandler
import logging
import sys

faulthandler.enable()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

from src.pipeline_builder import PipelineBuilder
from src.common import load_config


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline with Auto-Discovery")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Run multi-branch DeepStream pipeline")
    parser.add_argument("--config", default="configs/multi-branch.yaml", help="Config path")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[Config] Loaded: {args.config}")

    builder = PipelineBuilder(config)
    builder.build()
    builder.start_api()
    builder.wait_and_shutdown()

    print("[Done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
