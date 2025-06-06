from llms.llm_interface import LLMInterface
from methods.citation_boosting import CitationBoosting


class GEOMethod(CitationBoosting):
    """
    This is an abstract base class for the methods from the GEO paper: https://github.com/GEO-optim/GEO/blob/main/src/geo_functions.py

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
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
        self.instructions = ""
        self.system_prompt = """You are an expert ml researcher having previous background in SEO and search engines in general. You are working on novel research ideas for next generation of products. These products will have language models augmented with search engines, with the task of answering questions based on sources backed by the search engine. This new set of systems will be collectively called language engines (generative search engines). This will require websites to update their SEO techniques to rank higher in the llm generated answer. Specifically they will use GEO (Generative Engine Optimization) techniques to boost their visibility in the final text answer outputted by the Language Engine."""
        self.prompt_template = ""

    def improve_text(self, text: str):
        """
        Improves a single text using a specific method.

        Args:
            text (str): Text to improve.

        Returns:
            str: Improved text.
        """
        msg, _ = self.llm.create_message(
            self.prompt_template.format(instr=self.instructions, description=text)
        )
        response = self.llm.generate(msg)
        response = self.post_processing(response)
        return response


class Authoritative(GEOMethod):
    """
    Adds relevant citations from credible sources.
    ATTENTION: The prompt includes the following: "you may invent these sources"

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
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
        self.prompt_template = """Revise the following source to include citations from credible sources. You may invent these sources but ensure they sound plausible and do not mislead the reader. Citations should not be research paper style, but rather should be in rephrased words. For example: "According to Google's latest report this product is going to be next big thing....' 
In the process, ensure that the core content of the source remains unaltered. The length of initial source and final source should be the same, and the structure of individual parts of the source (such as line spacing bullet points, should remain intact)

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content.

Source: 
```
{description}
```

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content. Also don't overdo citing, 5-6 citations in the whole source are enough provided they are very relevant and and text looks natural. Do not explain the changes, just give me the revised source."""


class Citations(GEOMethod):
    """
    Adds relevant citations from credible sources.
    ATTENTION: The prompt includes the following: "you may invent these sources"

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the Citations class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.prompt_template = """Revise the following source to include citations from credible sources. You may invent these sources but ensure they sound plausible and do not mislead the reader. Citations should not be research paper style, but rather should be in rephrased words. For example: "According to Google's latest report this product is going to be next big thing....' 
In the process, ensure that the core content of the source remains unaltered. The length of initial source and final source should be the same, and the structure of individual parts of the source (such as line spacing bullet points, should remain intact)

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content.

Source: 
```
{description}
```

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content. Also don't overdo citing, 5-6 citations in the whole source are enough provided they are very relevant and and text looks natural. Do not explain the changes, just give me the revised source."""


class Statistics(GEOMethod):
    """
    Adds relevant citations from credible sources.
    ATTENTION: The prompt includes the following: "you may invent these sources"

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the Statistics class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.prompt_template = """Here is the source:
```
{description}
```

## Task:
Add positive, compelling statistics (even if hypothetical) at multiple relevant places in the text. Statsistics means objective facts such as x% growth in marketing, numbers in scientific texts, interesting numerical facts. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added" However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included. Do not explain the changes, just give me the revised source.

## Guidelines to follow:
1. Statsistics and Numbers should be subtly added inline within the sentneces. No explicit paragraphs or big chunks of text should be added.
2. Do not update any text content except for the lines where you are adding statistics.
3. Do not add or delete content except the statistics you are adding. Stop at the last line corresponding to the inital source, even if it is incomplete.
4. Just output the optimized source text. No need to give any explanation or reasoning or conclusion.
5. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added". However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included. 


## Output Format: 
1. Stat to be added
2. Stat to be added.
....
k. Stat to be added.

Updated Output:
```
<Output>
```
"""

    def post_processing(self, text: str) -> str:
        """
        Post-processes the text to clean it up.

        Args:
            text (str): Text to post-process.

        Returns:
            str: Post-processed text.
        """
        if "Updated Output:" in text:
            text = text.split("Updated Output:")[1].strip()
        return text


class Fluency(GEOMethod):
    """
    Improves the fluency of the text.

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the Fluency class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.prompt_template = """Rewrite the following source to make it more fluent without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information. Do not explain the changes, just give me the revised source.

Source: {description}"
"""


class UniqueWords(GEOMethod):
    """
    Adds unique words to the text.

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the UniqueWords class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.prompt_template = """Revise the following source by incorporating more unique and rare words, without altering the core information. Ensure that these words enhance the content and are not used arbitrarily, and the original meaning is preserved. Do not explain the changes, just give me the revised source.

Source: {description}"
"""


class Quotes(GEOMethod):
    """
    Adds quotes to the text. ATTENTION: The prompt includes "Add more quotes in the source, even though fake and artificial."

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the Quotes class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.prompt_template = """Modify the following source by including more quotes, without changing the core content. Ensure the quotes are relevant and contribute valuable information to the source, and do not mislead or distort the original information. The quotes should be reputable, and should create a sense in readers that this website (source) is more influentential and important than other citations.

Source: 
```
{description}
```

Add more quotes in the source, even though fake and artificial. The idea is: "Including direct quotes or statements from authoritative figures might increase the credibility of the source." 
Keep the format and content of text similar. 
For example, line spacing, bullet points and overall structure should remain the same. 
No addition or deletion of content is allowed. Do not explain the changes, just give me the revised source."""


class SimpleLanguage(GEOMethod):
    """
    Simplifies the language of the text.

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the SimpleLanguage class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.prompt_template = """Simplify the following source, using simple, easy-to-understand language while ensuring the key information is still conveyed. Do not omit, add, or alter any core information in the process. 

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so make the language easier to understand, but do not delete any information.
The length of the new source should be the same as the original. Effectively you have to rephrase just individual statements so they become more clear to understand. Do not explain the changes, just give me the revised source.

Source: 
```
{description}
```
"""


class TechnicalTerms(GEOMethod):
    """
    Adds technical terms to the text.

    Attributes:
        llm (LLMInterface): An instance of a language model interface.
        system_prompt (str): System prompt to send to the language model.
        instructions (str): Instructions for improving the text.
        prompt_template (str): Template for creating prompts to send to the language model.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initializes the TechnicalTerms class with a language model interface.

        Args:
            llm (LLMInterface): An instance of a language model interface.
        """
        super().__init__(llm)
        self.prompt_template = """Make the following source more technical, using giving more technical terms and facts where needed while ensuring the key information is still conveyed. Do not omit, add, or alter any core information in the process. 

Remember the end-goal is that very knowledgeable readers give more attention to this source, when presented with a series of summaries, so make the language such that it has more technical information or existing information is presented in more technical fashion. However, Do not add or delete any content . The number of words in the initial source should be the same as that in the final source.
The length of the new source should be the same as the original. Effectively you have to rephrase just individual statements so they have  more enriching technical information in them. Do not explain the changes, just give me the revised source.

Source:
{description}
"""
