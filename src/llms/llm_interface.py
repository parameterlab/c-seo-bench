from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMInterface(ABC):
    """
    This is an abstract base class for interacting with a Language Learning Model (LLM).
    Subclasses must implement all abstract methods to provide specific functionality.

    Usage:
        1. Create a message using `create_message`.
        2. Create a request using `create_request`.
        3. Run a batch of requests using `run_batch`.
        4. Retrieve results using `retrieve_results`.

    Methods:
        create_message(user_query: str, list_docs: List[Dict[str, Any]] = None) -> Any:
            Create a message based on user query and optional list of documents.

        create_request(messages: List[Dict[str, Any]], system: str, llm_model: str, i: int, max_tokens: int = 8192) -> Any:
            Create a request with the given parameters.

        run_batch(list_requests: List[Dict[str, Any]]) -> Any:
            Execute a batch of requests.

        retrieve_results(batch_id: str) -> Any:
            Retrieve results for a given batch ID.

        retrieve_text_response(response: Any) -> str:
            Extract text response from the LLM's response.

        get_citation_order(response: Any) -> List[int]:
            Get the order of citations from the LLM's response.

        calculate_api_call_cost(response: Any, input_cost: float, output_cost: float) -> float:
            Calculate the cost of an API call based on the response and given costs.
    """

    @abstractmethod
    def create_message(
        self, user_query: str, list_docs: List[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a message based on user query and optional list of documents.
        """
        pass

    @abstractmethod
    def generate(self, messages: List) -> Any:
        """
        Execute a single request.
        """
        pass

    @abstractmethod
    def create_request(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        i: int,
        max_tokens: int = 8192,
    ) -> Any:
        """
        Create a request with the given parameters.
        """
        pass

    @abstractmethod
    def run_batch(self, list_requests: List[Dict[str, Any]]) -> Any:
        """
        Execute a batch of requests.
        """
        pass

    @abstractmethod
    def retrieve_results(self, batch_id: str) -> Any:
        """
        Retrieve results for a given batch ID.
        """
        pass

    @abstractmethod
    def retrieve_text_response(self, response: Any) -> str:
        """
        Extract text response from the LLM's response.
        """
        pass

    @abstractmethod
    def get_citation_order(self, response: Any) -> List[int]:
        """
        Get the order of citations from the LLM's response.
        """
        pass

    @abstractmethod
    def calculate_response_cost(self, response: Any) -> float:
        """
        Calculate the cost of an API call based on the response.
        """
        pass

    @abstractmethod
    def calculate_batch_cost(self, responses: List[Any]) -> float:
        """
        Calculate the cost of a batch of API calls based on the responses.
        """
        pass

    @abstractmethod
    def get_error_messages(self, batch_id: str) -> List[str]:
        """
        Get error messages from the LLM's response.
        """
        pass

    @abstractmethod
    def get_status(self, batch_id: str) -> str:
        """
        Get the status of a batch request.
        """
        pass
