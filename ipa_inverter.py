import argparse
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache


# llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatOpenAI(model="gpt-4o")
# llm = ChatOpenAI(model="o1-mini") # tools not allowed for this model
llm = ChatOpenAI(model="gpt-4o-2024-08-06")


set_llm_cache(InMemoryCache())

# Step 1: IPA Transcriptions
ipa_schema = {
    "title": "ipa_transcriptions",
    "description": "Possible IPA transcriptions for a word.",
    "type": "object",
    "properties": {
        "ipa_transcriptions": {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": "^/.+/$"  # Ensures the transcription starts and ends with /
            }
        }
    },
    "required": ["ipa_transcriptions"]
}

ipa_prompt = PromptTemplate(
    input_variables=["word"],
    template="""
You are a phonetician.

Given the word "{word}", provide possible IPA transcriptions for the word.

Provide the transcriptions as a JSON object in the following format:
{{"ipa_transcriptions": ["transcription1", "transcription2"]}}

Example:
word: asprin
{{ "ipa_transcriptions": ["/ˈæs.pɹɪn/", "/ˈæs.pɚ.ɪn/"]}}
"""
)

structured_ipa_llm = llm.with_structured_output(ipa_schema)
ipa_chain = ipa_prompt | structured_ipa_llm

def get_ipa_transcriptions(word):
    input = {"word": word}
    response = ipa_chain.invoke(input)
    return response["ipa_transcriptions"]

# Step 2: Return to Orthography
orthography_schema = {
    "title": "spellings",
    "description": "Possible spellings for an a IPA transcription.",
    "type": "object",
    "properties": {
        "spellings": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["spellings"]
}

orthography_prompt = PromptTemplate(
    input_variables=["ipa_transcription"],
    template="""
    You are an expert in English orthography.

    Pretend that you are a human first hearing the following IPA transcription: "{ipa_transcription}".

    Given this transcription, list all possible spellings that this human might use to represent this pronunciation in English orthography after hearing it for the first time.
    
    Provide the spellings as a JSON object in the following format:
    {{"spellings": ["spelling1", "spelling2"]}}

    Example:
    transcription: "/ˈæs.pɹɪn/"
    {{"spellings": ["asprin", "asprine","asprinn","aspryn","assprin"]}}
    """
)

structured_orthography_llm = llm.with_structured_output(orthography_schema)
orthography_chain = orthography_prompt | structured_orthography_llm

def get_spellings(ipa_transcription):
    input = {"ipa_transcription": ipa_transcription}
    response = orthography_chain.invoke(input)
    return response["spellings"]

def get_misspellings(word:str):
    ipa_transcriptions = get_ipa_transcriptions(word)
    print("IPA transcriptions: ",ipa_transcriptions)
    spellings = []
    for ipa_transcription in ipa_transcriptions:
        spellings.extend(spelling.lower() for spelling in get_spellings(ipa_transcription))
    return spellings

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("word",type=str)
    args = argparser.parse_args()
    misspellings = set(get_misspellings(args.word))
    # new_misspellings = set()
    # for misspelling in misspellings:
    #     new_misspellings.add(misspelling)
    # for misspelling in misspellings:
    #     additional_misspellings = get_misspellings(misspelling) 
    #     for misspelling in additional_misspellings:
    #         new_misspellings.add(misspelling)
    # print(new_misspellings - misspellings)
    print("Spelling variations for",args.word,":",misspellings)
    # print(new_misspellings)
