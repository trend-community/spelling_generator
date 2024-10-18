# Step 4: end-to-end
import argparse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o")

misspelling_schema = {
    "title": "possible_misspellings",
    "description": "Possible misspellings for a word.",
    "type": "object",
    "properties": {
        "misspellings": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["misspellings"]
}

misspelling_prompt = PromptTemplate(
    input_variables=["word"],
    template="""
You are an expert in English orthography.

Given a word "{word}", list all possible misspellings of this word.

Provide the spellings as a JSON object in the following format:
{{"misspellings": ["misspelling1", "misspelling2"]}}

Example:
word: "aspirin"
{{"misspellings": ["aspirine", "asprin", "asprine","aspprin"]}}
""",
)
structured_misspelling_llm = llm.with_structured_output(misspelling_schema)
mispelling_chain = misspelling_prompt | structured_misspelling_llm
def get_possible_misspellings(word):
    input = {"word": word}
    response = mispelling_chain.invoke(input)
    return response["misspellings"]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("word",type=str)
    args = argparser.parse_args()
    misspellings = set(get_possible_misspellings(args.word))
    print(misspellings)
    