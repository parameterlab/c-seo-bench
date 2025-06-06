import itertools
import math
import os
import re

import numpy as np
import pandas as pd

from data import Benchmark
from llms import LLMInterface


class Engine:
    """
    The Engine class is responsible for running benchmarks on datasets using a specified LLMInterface.
    It processes the dataset, creates requests, saves input data, and executes the requests in batches.
    """

    def run_benchmark(
        self,
        dataset: Benchmark,
        developer_prompt: str,
        llm: LLMInterface,
        running_folder: str,
    ):
        """
        Runs a benchmark on the provided dataset using the specified LLMInterface.

        Args:
            dataset (Benchmark): The dataset to benchmark, containing user prompts and metadata.
            developer_prompt (str): The system-level prompt to guide the LLM's behavior.
            llm (LLMInterface): The LLM interface used to generate messages and run requests.
            running_folder (str): The folder where intermediate and output data will be stored.

        Returns:
            str: The batch ID of the executed requests.
        """
        list_columns = [
            "Prompt",
            "Response",
            "Search Query",
            "Boost Product Index",
            "Citation Order",
        ]
        list_rows = []

        # Create the requests
        list_requests = []
        for i, x in enumerate(dataset):
            if "search_results" in x:
                msg, raw_msg = llm.create_message(
                    x["user_prompt"], list_docs=x["search_results"]
                )
            else:
                msg, raw_msg = llm.create_message(x["user_prompt"])
            list_requests.append(
                llm.create_request(
                    msg,
                    developer_prompt,
                    i,
                )
            )
            raw_prompt = f"System: {developer_prompt}\n\n{raw_msg}"
            list_rows.append([raw_prompt, "", x["query"], x["boosted_indices"], None])

        # Save the input data
        df = pd.DataFrame(list_rows, columns=list_columns)
        os.makedirs(running_folder, exist_ok=True)
        df.to_parquet(os.path.join(running_folder, "requests.parquet"))

        # Run the requests
        batch_id = llm.run_batch(list_requests, running_folder)
        return batch_id

    def get_citation_order(self, text):
        # Regular expression to find numbers inside square brackets
        pattern = r"\[(\d+)\]"
        # Find all matches in the text
        matches = re.findall(pattern, text)
        # Convert matches to integers
        citations_w_dups = [
            int(match) - 1 for match in matches
        ]  # subtract 1 to convert to 0-based index
        # remove duplicates preserving order
        citations = list(dict.fromkeys(citations_w_dups))
        return citations, citations_w_dups

    def process_benchmark_responses(self, responses_txt, output_folder):
        running_folder = output_folder.replace("results", "running")
        df = pd.read_parquet(os.path.join(running_folder, "requests.parquet"))

        list_citation_orders = []
        list_citation_orders_w_dups = []
        cnt_errors = 0

        for idx, response in enumerate(responses_txt):
            try:
                citations, citations_w_dups = self.get_citation_order(response)

            except Exception as e:
                print(f"Error in index {idx}")
                citations = []
                citations_w_dups = []
                cnt_errors += 1
            list_citation_orders.append(citations)
            list_citation_orders_w_dups.append(citations_w_dups)

        df["Response"] = responses_txt
        df["Citation Order"] = list_citation_orders
        df["Citation Order w. Duplicates"] = list_citation_orders_w_dups

        print(f"Errors in {cnt_errors} out of {len(df)}")
        return df
