import json
import random

import requests
from datasets import load_dataset

from config.adoption_mode import AdoptionMode


class Benchmark:
    """
    Benchmark class for evaluating Contextual-SEO (C-SEO) methods.

    This class loads a dataset and a set of selected/improved documents, then prepares
    user prompts and search result contexts for benchmarking various document ranking or
    rewriting strategies (SEO and C-SEO methods). It supports different boosting
    and ranking methods, and can generate deterministic data points for evaluation.
    """

    def __init__(
        self,
        num_docs_in_context=10,
        method="baseline",
        sample_size=None,
        data_path="cseo/cseo-bench",
        split="retail",
        doc_type="document",
        selected_documents_path=None,
    ):
        """
        Initializes the Benchmark class.

        Args:
            num_docs_in_context (int): Number of documents to include in context. Defaults to 10.
            method (str): The method used for boosting or baseline. Defaults to "baseline".
            sample_size (int, optional): Number of queries to sample. If None, all queries are used. Defaults to None.
            data_path (str): Path or identifier for the dataset. Defaults to "cseo/cseo-bench".
            split (str): Dataset split to use (e.g., "nq_snippets"). Defaults to "retail".
            doc_type (str): Type of document (e.g., product, game, news article). Used in user prompts.
            selected_documents_path (str): Path to the selected documents JSON file.
        """
        self.num_docs_in_context = num_docs_in_context
        self.data_path = data_path
        self.split = split
        self.name = split
        self.method = method
        self.doc_type = doc_type
        print(f"Loading Benchmark - {split} dataset...")
        ds = load_dataset(
            self.data_path, split=split
        )  # , download_mode="force_redownload"
        # setting main components of the object
        self.df = ds.to_pandas()
        self.query_ids = self.df["query_id"].unique()

        if selected_documents_path is not None:
            # Load selected documents from a JSON file
            with open(selected_documents_path, "r", encoding="utf-8") as f:
                self.selected_docs = json.load(f)
        else:
            if self.method != "baseline":
                raise ValueError(
                    "Selected documents must be provided for non-baseline methods."
                )
            self.selected_docs = {}

        if sample_size:
            self.query_ids = self.query_ids[:sample_size]
        self.list_data_points = self.preload_data()
        print(f"{self.split} dataset loaded.")

    def preload_data(self):
        """
        Preloads all data points.

        Returns:
            list: List of data points, each as a dictionary with user prompt, query, boosted indices, and documents.
        """
        list_data_points = []
        for idx in range(len(self)):
            list_data_points.append(self.data_point_docs_in_context(idx))
        return list_data_points

    def data_point_docs_in_context(self, idx):
        """
        Generates a user query string with search results for a given query index.

        Args:
            idx (int): Index of the query.

        Returns:
            dict: Dictionary with keys:
                - user_prompt (str): The formatted user prompt with search results.
                - query (str): The query string.
                - boosted_indices (list): List of indices of boosted documents.
                - list_docs (list): List of document strings in context.
        """
        query_id = self.query_ids[idx]
        hits = self.df[self.df["query_id"] == query_id][: self.num_docs_in_context]
        query = hits["query"].values[0]
        try:
            search_results, list_docs, boost_list = self.search_results_string(
                hits, idx
            )
        except Exception as e:
            print(f"Error in index {idx}")
            raise e
        user_query = f"Question: {query}\n\n" f"Search Results:\n{search_results}"
        return {
            "user_prompt": user_query,
            "query": query,
            "boosted_indices": boost_list,
            "list_docs": list_docs,
        }

    def search_results_string(self, tag_hits, idx):
        """
        Generates a formatted string of search results and identifies boosted indices.

        Args:
            tag_hits (pd.DataFrame): DataFrame rows for the current query.
            idx (int): Index of the query.

        Returns:
            tuple:
                - search_results (str): Formatted string of search results.
                - list_docs (list): List of document strings in order.
                - boost_list (list): Sorted list of boosted document indices (0-based).
        """
        if self.method == "baseline":
            boost_set = []
        else:
            boost_set = [int(x) for x in self.selected_docs[str(idx)].keys()]

        # New behavior: if seo_baseline-idx, promote the target doc into idx position
        if self.method.startswith("seo_baseline-") and boost_set:
            return self.__seo_baseline_at_position_i(tag_hits, boost_set)
        elif self.method == "seo_baseline_game_theory":
            return self.__seo_baseline(tag_hits, boost_set)
        else:
            # C-SEO Methods
            search_results = ""
            list_docs = []
            for i, (_, hit) in enumerate(tag_hits.iterrows()):
                if i in boost_set and self.method != "baseline":
                    doc = self.selected_docs[str(idx)][str(i)][f"{self.method}(doc)"]
                else:
                    doc = hit["document"]
                search_results += (
                    f"{self.doc_type} {i+1}:\n{doc}\n\n##########################\n\n"
                )
                list_docs.append(doc)
            return search_results, list_docs, sorted(list(boost_set))

    def __seo_baseline(self, tag_hits, boost_set):
        """
        Promotes all documents in boost_set to the top positions in the search results.

        Args:
            tag_hits (pd.DataFrame): DataFrame rows for the current query.
            boost_set (list): List of indices to promote.

        Returns:
            tuple:
                - search_results (str): Formatted string of search results.
                - list_docs (list): List of document strings in new order.
                - sorted(boost_set) (list): Sorted list of boosted indices.
        """
        # put all docs from boost_set (indexes) in the first positions
        # and the rest of the docs in the rest of the positions
        n = len(tag_hits)
        list_docs = []
        for i in range(n):
            hit = tag_hits.iloc[i]
            doc = hit["document"]
            list_docs.append(doc)
        # put all docs from boost_set (indexes) in the first positions
        promoted_idx = 0
        for i in sorted(list(boost_set)):
            doc = tag_hits.iloc[i]["document"]
            list_docs.insert(promoted_idx, list_docs.pop(i))
            promoted_idx += 1
        # Now, list_docs contains the documents in the desired order
        search_results = ""
        for i, doc in enumerate(list_docs):
            search_results += (
                f"{self.doc_type} {i+1}:\n{doc}\n\n##########################\n\n"
            )
        return search_results, list_docs, sorted(list(boost_set))

    def __seo_baseline_at_position_i(self, tag_hits, boost_set):
        """
        Promotes a single target document to a specified position in the search results. The position is determined by the method name (e.g., "seo_baseline-3" promotes to position 3).
        This method assumes that the method name is in the format "seo_baseline-<idx>", where <idx> is the 1-based index of the position to promote the document to.

        Args:
            tag_hits (pd.DataFrame): DataFrame rows for the current query.
            boost_set (list): List containing the index of the document to promote.

        Returns:
            tuple:
                - search_results (str): Formatted string of search results.
                - list_docs (list): List of document strings in new order.
                - sorted(boost_set) (list): Sorted list of boosted indices.
        """
        n = len(tag_hits)
        target_doc_idx = sorted(list(boost_set))[0]  # 0-based index
        # Promote the target doc to the idx position
        # and shift the rest of the documents
        promoted_idx = (
            int(self.method.split("-")[1]) - 1
        )  # -1 to convert to 0-based index
        # I need to swap the idx position with the target doc

        list_docs = []
        for i in range(n):
            hit = tag_hits.iloc[i]
            doc = hit["document"]
            list_docs.append(doc)
        list_docs.insert(promoted_idx, list_docs.pop(target_doc_idx))

        search_results = ""
        for i, doc in enumerate(list_docs):
            search_results += (
                f"{self.doc_type} {i+1}:\n{doc}\n\n##########################\n\n"
            )

        return search_results, list_docs, sorted(list(boost_set))

    def __len__(self):
        """
        Returns the number of unique queries in the benchmark.

        Returns:
            int: Number of queries.
        """
        return len(self.query_ids)

    def __getitem__(self, idx):
        """
        Retrieves the preloaded data point at the specified index.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: Data point dictionary.
        """
        return self.list_data_points[idx]

    def __iter__(self):
        """
        Iterates over all preloaded data points.

        Yields:
            dict: Data point dictionary for each query.
        """
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def get_name(self):
        """
        Returns the name of the benchmark (usually the split name).

        Returns:
            str: Name of the benchmark.
        """
        return self.name
