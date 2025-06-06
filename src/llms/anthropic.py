import getpass
import json
import os
from typing import List

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .llm_interface import LLMInterface


class AnthropicHelper(LLMInterface):
    def __init__(self, llm_name: str):
        """
        Initialize the AnthropicHelper class with an API key.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = getpass.getpass(
                "Enter API key for Claude: "
            )
        api_key = os.environ["ANTHROPIC_API_KEY"]
        self.llm_name = llm_name
        self.client = anthropic.Anthropic(api_key=api_key)

        self.PRICES = {
            "claude-3-7-sonnet-20250224": {
                "input": 3.0,
                "output": 15.0,
                "prompt_caching_write": 3.75,
                "prompt_caching_read": 0.30,
            },
            "claude-3-5-sonnet-20241022": {
                "input": 3.0,
                "output": 15.0,
                "prompt_caching_write": 3.75,
                "prompt_caching_read": 0.30,
            },
            "claude-3-5-haiku-20241022": {
                "input": 0.80,
                "output": 4.0,
                "prompt_caching_write": 1.0,
                "prompt_caching_read": 0.10,
            },
            "claude-3-opus-20240314": {
                "input": 15.0,
                "output": 75.0,
                "prompt_caching_write": 18.75,
                "prompt_caching_read": 1.50,
            },
        }

    def create_message(self, user_query, list_docs=None):
        """
        Create a message for the Anthropic API.

        Args:
            user_query (str): The user's query.
            list_docs (list, optional): List of documents to be included in the message.

        Returns:
            tuple: A tuple containing the messages, and raw prompt.
        """
        list_anthropic_docs = []
        if list_docs is not None:
            for doc in list_docs:
                anthropic_doc = {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [
                            {
                                "type": "text",
                                "text": doc["doc"],
                            }
                        ],
                    },
                    "title": doc["title"],  # optional
                    "citations": {"enabled": True},
                }

                list_anthropic_docs.append(anthropic_doc)

        content = list_anthropic_docs + [{"type": "text", "text": user_query}]
        messages = [{"role": "user", "content": content}]
        raw_prompt = "User: " + user_query
        return messages, raw_prompt

    def create_request(self, messages, system, i, max_tokens=8192):
        """
        Create a request for the Anthropic API.

        Args:
            messages (list): List of messages to be included in the request.
            system (str): The system prompt.
            llm_model (str): The LLM model to be used.
            i (int): The request index.
            max_tokens (int, optional): The maximum number of tokens. Default is 8192.

        Returns:
            Request: The request object.
        """
        request = Request(
            custom_id=f"request-{i}",
            params=MessageCreateParamsNonStreaming(
                model=self.llm_name,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system}],
                messages=messages,
            ),
        )
        return request

    def generate(self, messages):
        raise NotImplementedError(
            "The generate method is not implemented in the AnthropicHelper class."
        )

    def run_batch(self, list_requests: List[Request], output_folder):
        """
        Run a batch of requests.

        Args:
            list_requests (List[Request]): List of Request objects to be processed.

        Returns:
            object: The response from the batch create API call.
        """
        # 1) Save batch requests as JSONL format (required by OpenAI API)
        batch_filename = os.path.join(output_folder, "requests.jsonl")
        with open(batch_filename, "w", encoding="utf-8") as f:
            for request in list_requests:
                f.write(json.dumps(request) + "\n")

        batch_response_id = self.client.messages.batches.create(
            requests=list_requests
        ).id

        with open(
            os.path.join(output_folder, "metadata.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps({"batch_response_id": batch_response_id}) + "\n")

        return batch_response_id

    def retrieve_results(self, batch_id):
        """
        Retrieve the results of a batch request.

        Args:
            batch_id (str): The ID of the batch request.

        Returns:
            object: The response object if processing is complete, otherwise None.
        """
        status = self.client.messages.batches.retrieve(batch_id)
        request_counts = status.request_counts
        total_requests_num = sum(request_counts.model_dump().values())
        if status.processing_status == "ended":
            num_errors = request_counts.errored
            if num_errors > 0:
                print(f"Number of errors: {num_errors}. Saving successful results.")
            # results is a .jsonl file. It has one response line for every successful request line in the input file.
            list_responses = [x for x in self.client.messages.batches.results(batch_id)]
            # The results might not be in the same order as the requests. That's why we assinged custom_id to each request.
            # sort them by custom_id
            sorted_results = [None] * total_requests_num
            for response in list_responses:
                i = int(response.custom_id.split("-")[-1])
                sorted_results[i] = self.retrieve_text_response(
                    response.result.message
                )  # sort by custom_id
            cost = self.calculate_batch_cost(list_responses)
            return sorted_results, cost
        else:
            print("Batch not completed yet")
            return None, None

    def retrieve_text_response(self, response):
        """
        Retrieve the anthropic text response from the given response object.

        Args:
            response (object): The response object containing the anthropic text.

        Returns:
            str: The concatenated anthropic text from the response.
        """
        text = ""
        for content in response.content:
            text += content.text
            if content.citations is not None:
                for citation in content.citations:
                    text += f" [{citation.document_index}]"
        return text

    def get_citation_order(self, response):
        """
        Get the order of citations in the response.

        Args:
            response (object): The response object containing the citations.

        Returns:
            list: The order of citations in the response.
        """
        citations = []
        for content in response.content:
            if content.citations is not None:
                for citation in content.citations:
                    citations.append(citation.document_index)
        return citations

    def get_citation_order_from_batch(self, batch_id):
        list_responses = [x for x in self.client.messages.batches.results(batch_id)]
        list_citation_orders = []
        list_citation_orders_w_dups = []
        for response in list_responses:
            list_citation_orders_w_dups.append(
                self.get_citation_order(response.result.message)
            )
            citations = list(dict.fromkeys(list_citation_orders_w_dups[-1]))
            list_citation_orders.append(citations)
        return list_citation_orders, list_citation_orders_w_dups

    def calculate_response_cost(self, response):
        """
        Calculate the cost of an Anthropic API call.

        Args:
            response (object): The response object containing usage information.

        Returns:
            float: The total cost of the API call.
        """
        input_cost = response.usage.input_tokens * (
            self.PRICES[self.llm_name]["input"] / 1e6
        )
        output_cost = response.usage.output_tokens * (
            self.PRICES[self.llm_name]["output"] / 1e6
        )
        return input_cost + output_cost

    def calculate_batch_cost(self, list_responses):
        total_cost = 0
        for response in list_responses:
            input_cost = response.result.message.usage.input_tokens * (
                (self.PRICES[self.llm_name]["input"] / 2) / 1000000
            )
            output_cost = response.result.message.usage.input_tokens * (
                (self.PRICES[self.llm_name]["output"] / 2) / 1000000
            )
            cache_creation_input_cost = response.result.message.usage.input_tokens * (
                (self.PRICES[self.llm_name]["prompt_caching_write"] / 2) / 1000000
            )
            cache_read_input_tokens = response.result.message.usage.input_tokens * (
                (self.PRICES[self.llm_name]["prompt_caching_read"] / 2) / 1000000
            )
            cost = (
                input_cost
                + output_cost
                + cache_creation_input_cost
                + cache_read_input_tokens
            )
            total_cost += cost
        return total_cost

    def get_error_messages(self, batch_id):
        return super().get_error_messages(batch_id)

    def get_status(self, batch_id):
        return self.client.messages.batches.retrieve(batch_id).processing_status
