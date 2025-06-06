from llms.llm_interface import LLMInterface
from methods.citation_boosting import CitationBoosting


class LLMstxt(CitationBoosting):
    """ """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the LLMstxt class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.instructions = """Create a llms.txt markdown file to provide LLM-friendly content. This file summarizes the main text and offers brief background information, guidance, and links (if available).

Follow this template
# Title

> Introduction paragraph

Optional details go here

## Section name
More details

Here is the content of the text:\n{text}"""
