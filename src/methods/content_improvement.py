from llms.llm_interface import LLMInterface
from methods.citation_boosting import CitationBoosting


class ContentImprovement(CitationBoosting):
    """
    A class to improve the fluency, authority, and persuasiveness of texts using a language model.
    This is an ethical method aiming to improve the quality of the text without altering the core content.

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the ContentImprovement class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.instructions = "Rewrite the following text to make it more fluent, authoritative, and persuasive without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information. The revised text should reflect confidence, expertise, and assertiveness while maintaining the original content's meaning and relevance. The text should be assertive in its statements, such that the reader believes that this is a more valuable source of information than other texts. Lastly, give structure to the text."
