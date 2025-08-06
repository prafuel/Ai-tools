from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from src.config.config import load_groq, load_mistral, load_huggingface
from langchain.output_parsers import PydanticOutputParser

# ==================================================================== #
class ComplexWordMap(BaseModel):
    complex_word_map: dict[str, str] = Field(
        description="A mapping of complex words to their simpler alternatives that maintain the same meaning. Format: 'complex_word': 'simple_replacement'"
    )
    updated_sentence: str = Field(
        description="Using word map replace it in main sentence"
    )

parser = PydanticOutputParser(pydantic_object=ComplexWordMap)
# ==================================================================== #

prompt = PromptTemplate(
    template="""
    You are a text simplification expert. Your task is to identify complex or difficult words in the given text and provide simpler alternatives that maintain the original meaning and context.

    Instructions:
    1. Analyze the provided text carefully
    2. Identify words that might be considered complex, advanced, or difficult for general readers
    3. For each complex word, provide a simpler alternative that:
       - Maintains the same meaning in context
       - Is more commonly understood
       - Fits grammatically in the original sentence
    4. Focus on words that would genuinely benefit from simplification
    5. Do not replace words that are already simple or commonly used

    Examples of good replacements:
    - "melancholy" → "sad"
    - "utilize" → "use"
    - "facilitate" → "help"
    - "demonstrate" → "show"
    - "substantial" → "large"

    Text to analyze: "{text}"

    Provide your response in the following format:
    {format_instructions}
    """,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ==================================================================== #

# Chain building
# groq_chain = prompt | load_groq("gemma2-9b-it") | parser

# mistral_chain = prompt | load_mistral("mistral-large2411-fai-doc") | parser

mistral_chain = prompt | load_huggingface() | parser
# ==================================================================== #
response = mistral_chain.invoke(input={"text": "Despite his ostentatious demeanor, her skepticism remained unshaken"})
print(response)
