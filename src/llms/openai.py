import getpass
import json
import os

from openai import OpenAI
from llms.llm_interface import LLMInterface


class OpenAIHelper(LLMInterface):
    """
    A helper class to interact with the OpenAI API for batch processing.

    Attributes:
        client (OpenAI): The OpenAI client instance.
    """

    def __init__(self, llm_name: str):
        """
        Initializes the OpenAIHelper with an OpenAI client instance.
        Arguments:
            llm_name {str} -- The name of the LLM model.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        self.llm_name = llm_name
        self.client = OpenAI()

        self.STANDARD_PRICES = {
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
            "gpt-4o-2024-11-20": {"input": 2.5, "output": 10.0},
            "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
            "gpt-4o-audio-preview-2024-12-17": {"input": 2.5, "output": 10.0},
            "gpt-4o-audio-preview-2024-10-01": {"input": 2.5, "output": 10.0},
            "gpt-4o-realtime-preview-2024-12-17": {"input": 5.0, "output": 20.0},
            "gpt-4o-realtime-preview-2024-10-01": {"input": 5.0, "output": 20.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
            "gpt-4o-mini-audio-preview-2024-12-17": {"input": 0.15, "output": 0.6},
            "gpt-4o-mini-realtime-preview-2024-12-17": {"input": 0.6, "output": 2.4},
            "o1": {"input": 15.0, "output": 60.0},
            "o1-2024-12-17": {"input": 15.0, "output": 60.0},
            "o1-preview-2024-09-12": {"input": 15.0, "output": 60.0},
            "o3-mini": {"input": 1.1, "output": 4.4},
            "o3-mini-2025-01-31": {"input": 1.1, "output": 4.4},
            "o1-mini": {"input": 1.1, "output": 4.4},
            "o1-mini-2024-09-12": {"input": 1.1, "output": 4.4},
        }

        self.BATCH_PRICES = {
            "gpt-4o": {"input": 1.25, "output": 5.0},
            "gpt-4o-2024-08-06": {"input": 1.25, "output": 5.0},
            "gpt-4o-2024-11-20": {"input": 1.25, "output": 5.0},
            "gpt-4o-2024-05-13": {"input": 2.5, "output": 7.5},
            "gpt-4o-mini": {"input": 0.075, "output": 0.3},
            "gpt-4o-mini-2024-07-18": {"input": 0.075, "output": 0.3},
            "o1": {"input": 7.5, "output": 30.0},
            "o1-2024-12-17": {"input": 7.5, "output": 30.0},
            "o1-preview-2024-09-12": {"input": 7.5, "output": 30.0},
            "o3-mini": {"input": 0.55, "output": 2.2},
            "o3-mini-2025-01-31": {"input": 0.55, "output": 2.2},
            "o1-mini": {"input": 0.55, "output": 2.2},
            "o1-mini-2024-09-12": {"input": 0.55, "output": 2.2},
        }

    def create_message(self, user_query, list_docs=None):
        """
        Creates a message payload for the OpenAI API.

        Args:
            user_query (str): The user's query.
            list_docs (list, optional): A list of documents to include in the message.

        Returns:
            tuple: A tuple containing the messages list and the raw prompt string.
        """
        messages = [
            {
                "role": "user",
                "content": user_query,
            },
        ]

        raw_prompt = "User: " + user_query
        return messages, raw_prompt

    def generate(self, messages, response_format=None):
        """
        Generates a response from the OpenAI API based on the provided messages.

        Args:
            messages (list): A list of message dictionaries.
            response_format (str, optional): The desired response format.

        Returns:
            dict: The generated response message.
        """
        if response_format is None:
            # response format is only available in new models
            completion = self.client.chat.completions.create(
                model=self.llm_name, messages=messages
            )
        else:
            if response_format == "json":
                response_format = {"type": "json_object"}
            completion = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                response_format=response_format,
            )
        cost = self.calculate_response_cost(completion)
        return completion.choices[0].message, cost

    def create_request(
        self,
        messages,
        system,
        i,
        max_completion_tokens=8192,
        reasoning_effort=None,
    ):
        """
        Creates a request payload for the OpenAI API.

        Args:
            messages (list): A list of message dictionaries.
            system (str): The system message content.
            i (int): The request index.
            max_completion_tokens (int, optional): The maximum number of completion tokens.
            reasoning_effort (str, optional): The reasoning effort parameter.

        Returns:
            dict: The request payload dictionary.
        """
        if system != "" or system is not None:
            messages = [{"role": "developer", "content": system}] + messages
        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.llm_name,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
            },
        }
        if "o1" in self.llm_name or "o3" in self.llm_name:
            request["body"]["reasoning_effort"] = reasoning_effort
        return request

    def run_batch(self, list_requests, output_folder):
        """
        Runs a batch of OpenAI API requests and saves the response id.

        Args:
            list_requests (list): A list of request dictionaries.
            output_folder (str): The folder to save the batch requests and metadata.

        Returns:
            str: The ID of the created batch response.
        """
        # 1) Save batch requests as JSONL format (required by OpenAI API)
        batch_filename = os.path.join(output_folder, "requests.jsonl")
        with open(batch_filename, "w", encoding="utf-8") as f:
            for request in list_requests:
                f.write(json.dumps(request) + "\n")

        batch_input_file = self.client.files.create(
            file=open(batch_filename, "rb"), purpose="batch"
        )

        # save batch input file id
        batch_input_file_id = batch_input_file.id
        with open(
            os.path.join(output_folder, "metadata.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps({"batch_input_file_id": batch_input_file_id}) + "\n")

        # 2) Create a batch job
        batch_input_file_id = batch_input_file.id
        batch_response = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": output_folder},
        )
        batch_response_id = batch_response.id
        with open(
            os.path.join(output_folder, "metadata.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps({"batch_response_id": batch_response_id}) + "\n")

        return batch_response_id

    def retrieve_results(self, batch_response_id):
        """
        Retrieves the results of a completed batch job.

        Args:
            batch_response_id (str): The ID of the batch response.

        Returns:
            tuple or None: A tuple containing a list of sorted result dictionaries and the total cost if the batch is completed, otherwise None.
        """
        status = self.client.batches.retrieve(batch_response_id)
        if status.status == "completed":
            num_errors = status.request_counts.failed
            if num_errors > 0:
                print(f"Number of errors: {num_errors}. Saving successful results.")
            # results is a .jsonl file. It has one response line for every successful request line in the input file.
            list_results = get_json_list(
                self.client.files.content(status.output_file_id).text
            )
            # The results might not be in the same order as the requests. That's why we assinged custom_id to each request.
            # sort them by custom_id
            sorted_results = [None] * status.request_counts.total
            for result in list_results:
                i = int(result["custom_id"].split("-")[-1])
                sorted_results[i] = self.retrieve_text_response(
                    result
                )  # sort by custom_id
            cost = self.calculate_batch_cost(list_results)
            return sorted_results, cost
        else:
            print("Batch not completed yet")
            return None, None

    def retrieve_text_response(self, response):
        """
        Retrieves the text content from a response.

        Args:
            response (dict): The response dictionary.

        Returns:
            str: The text content of the response.
        """
        return response["response"]["body"]["choices"][0]["message"]["content"]

    def get_citation_order(self, response):
        # Implement this method if needed for OpenAI
        pass

    def calculate_response_cost(self, response):
        """
        Calculates the cost of a single response.

        Args:
            response (dict): The response dictionary.

        Returns:
            float: The calculated cost of the response.
        """
        input_cost = (
            response.usage.prompt_tokens
            * self.STANDARD_PRICES[self.llm_name]["input"]
            / 1000000
        )
        completion_cost = (
            response.usage.completion_tokens
            * self.STANDARD_PRICES[self.llm_name]["output"]
            / 1000000
        )
        return input_cost + completion_cost

    def calculate_batch_cost(self, responses):
        """
        Calculates the total cost of a batch of responses.

        Args:
            responses (list): A list of response dictionaries.

        Returns:
            float: The total calculated cost of the batch.
        """
        input_price = self.BATCH_PRICES[self.llm_name]["input"]
        output_price = self.BATCH_PRICES[self.llm_name]["output"]
        total_cost = 0
        for response in responses:
            prompt_cost = response["response"]["body"]["usage"]["prompt_tokens"] * (
                input_price / 1000000
            )
            completion_cost = response["response"]["body"]["usage"][
                "completion_tokens"
            ] * (output_price / 1000000)
            total_cost += prompt_cost + completion_cost
        return total_cost

    def retrieve_openai_batch_responses(self, batch_response_id):
        """
        Retrieves the responses of a completed OpenAI batch job.

        Args:
            batch_response_id (str): The ID of the batch response to retrieve.

        Returns:
            list or None: A list of response dictionaries if the batch is completed, otherwise None.
        """
        if self.client.batches.retrieve(batch_response_id).status == "completed":
            batch = self.client.batches.retrieve(batch_response_id)
            file_response = self.client.files.content(batch.output_file_id)
            return get_json_list(file_response.text)
        else:
            print("Batch not completed yet")
            return None

    def get_error_messages(self, batch_id):
        """
        Retrieves error messages from a batch job.

        Args:
            batch_id (str): The ID of the batch job.

        Returns:
            list: A list of error messages.
        """
        status = self.client.batches.retrieve(batch_id)
        list_errors = get_json_list(
            self.client.files.content(status.error_file_id).text
        )
        return list_errors

    def get_status(self, batch_id):
        """
        Retrieves the status of a batch job.

        Args:
            batch_id (str): The ID of the batch job.

        Returns:
            str: The status of the batch job.
        """
        return self.client.batches.retrieve(batch_id).status


def get_json_list(jsonl_text):
    """
    Parses a JSONL text string into a list of JSON objects.

    Args:
        jsonl_text (str): The JSONL text string.

    Returns:
        list: A list of JSON objects.
    """
    return [json.loads(line) for line in jsonl_text.split("\n") if line]
