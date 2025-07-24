from dotenv import load_dotenv
load_dotenv(".env")
import os
import json
import random
import argparse
import logging
from datetime import datetime
from typing import Dict
from tqdm import tqdm

from moatless.benchmark.utils import get_moatless_instance
from moatless.benchmark.swebench import create_repository
from moatless.index.code_index import CodeIndex
from moatless.index.settings import IndexSettings
from moatless.file_context import FileContext


def main(instance_id):
    print(instance_id)
    instance = get_moatless_instance(split='verified',instance_id=instance_id)  # The instance obtained is a locally downloaded swe-bench with some modified attributes
    repository = create_repository(instance)
    code_index = CodeIndex.from_index_name(
        instance["instance_id"], file_repo=repository
    )
    file_context = FileContext(repo=repository)


if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Add instance_id parameter, can be a list or string, default is empty, required parameter
    # parser.add_argument("--instance_ids", nargs='+', default=[], required=True,
    #                     help="The instance ID(s), can be a list or a single string.")
    parser.add_argument("--instance_ids", type=str, required=True,
                        help="The file path to instance ID(s)")

    args = parser.parse_args()

    with open(args.instance_ids, "r", encoding='utf-8') as f:
        instance_ids = [line.strip() for line in f if line.strip()]

    if isinstance(instance_ids, list):
        for instance_id in tqdm(instance_ids, desc="Processing instances"):
            main(instance_id)
    elif isinstance(instance_ids, str):
        main(instance_ids)

    print('load finished')