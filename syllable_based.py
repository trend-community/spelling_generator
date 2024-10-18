
import argparse
import itertools
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-2024-08-06")

syllabify_schema = {
  "title": "Syllabification",
  "description": "Syllabification of a word.",
  "type": "object",
  "properties": {
    "syllables": {
      "type": "array",
      "items": { "type": "string"}
    }
  },
  "required": ["syllables"]
}


# Create a prompt template for syllabification
syllabify_prompt = PromptTemplate(
    input_variables=["word"],
    template="""
You are an expert linguist.

Given the word "{word}", transcribe it phonetically, and then break it down into its syllables.

Provide the syllables as a JSON object in the following format:
{{"syllables": ["syllable1", "syllable2", "syllable3"]}}

Example:
word: aspirin
{{"syllables": ["asp","ruhn"]}}
""",
)
# Create an LLMChain for syllabification with structured output
structured_llm = llm.with_structured_output(syllabify_schema)
syllabify_chain = syllabify_prompt | structured_llm

def syllabify_word(word:str)->list[str]:
    """
    Splits a given word into its constituent syllables.
    Args:
        word (str): The word to be syllabified.
    Returns:
        list: A list of syllables that make up the word.
    """

    input = {"word": word}
    response = syllabify_chain.invoke(input)
    # The response is already parsed according to the schema
    syllables = response["syllables"]
    return syllables

# Step 2: IPA Transcriptions
ipa_schema = {
    "title": "ipa_transcriptions",
    "description": "Possible IPA transcriptions for a syllable.",
    "type": "object",
    "properties": {
        "ipa_transcriptions": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["ipa_transcriptions"]
}

ipa_prompt = PromptTemplate(
    input_variables=["syllable","word"],
    template="""
You are a phonetician.

Given the syllable "{syllable}" in the word "{word}", list all possible IPA (International Phonetic Alphabet) transcriptions for it, considering different pronunciations.

Provide the transcriptions as a JSON object in the following format:
{{"ipa_transcriptions": ["transcription1", "transcription2"]}}

Example:
syllable: "to"
{{ "ipa_transcriptions": ["/toʊ/", "/tuː/", "/tə/"]}}
"""
)

structured_ipa_llm = llm.with_structured_output(ipa_schema)
ipa_chain = ipa_prompt | structured_ipa_llm

def get_ipa_transcriptions(syllable:str,word:str)->list[str]:
    """
        Retrieve IPA transcriptions for a given syllable and word.

        Args:
            syllable (str): The syllable to be transcribed, e.g. "as" or "pir" as in "aspirin".
            word (str): The word containing the syllable, e.g. "aspirin".

        Returns:
            list: A list of IPA transcriptions for the given syllable and word.
        """
    input = {"syllable": syllable,"word": word}
    response = ipa_chain.invoke(input)
    ipa_list = response["ipa_transcriptions"]
    return ipa_list

# Step 3: Possible Spellings
spelling_schema = {
    "title": "possible_spellings",
    "description": "Possible spellings for an IPA transcription.",
    "type": "object",
    "properties": {
        "spellings": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["spellings"]
}

spelling_prompt = PromptTemplate(
    input_variables=["ipa_transcription","syllable","word"],
    template="""
You are an expert in English orthography.

Given the IPA transcription "{ipa_transcription}" of the syllable "{syllable}" in the word "{word}", list all possible English spellings (including misspellings) of this syllable that correspond to this pronunciation.
These spellings will be concatenated to form a possible misspelling so do not include extraneous punctuation or spaces.

Provide the spellings as a JSON object in the following format:
{{"ipa_transcription": "{ipa_transcription}", "spellings": ["spelling1", "spelling2"]}}

Example:
ipa_transcription: "/ʌl/" 
{{"spellings": ["ul", "ull", "al","all"]}}
""",
)

structured_spelling_llm = llm.with_structured_output(spelling_schema)
spelling_chain = spelling_prompt | structured_spelling_llm


def get_possible_spellings(ipa_transcription,syllable,word):
    input = {"ipa_transcription": ipa_transcription,
             "syllable": syllable,
             "word": word}
    response = spelling_chain.invoke(input)
    spellings = response["spellings"]
    cleaned_spellings = [spelling for spelling in spellings if spelling.isalpha()]
    return cleaned_spellings



def get_syllable_based_misspellings(word):
    syllables = syllabify_word(word)
    print(f"Syllables of '{word}': {syllables}")
    syllable_to_ipa_transcription = {syllable: get_ipa_transcriptions(syllable,word) for syllable in syllables}
    get_spellings = lambda syllable,ipa_transcriptions: set(spelling for ipa_transcription in ipa_transcriptions for spelling in get_possible_spellings(ipa_transcription,syllable,word))

    syllable_to_alternative_spellings = {
      syllable:  get_spellings(syllable,ipa_transcriptions) | set([syllable]) for syllable,ipa_transcriptions in syllable_to_ipa_transcription.items()
  }
    print(syllable_to_alternative_spellings)
  #take cartesian product of all alternative syllable spellings
    alternative_spellings = ["".join(spelling) for spelling in itertools.product(*[spellings for spellings in syllable_to_alternative_spellings.values()])]
    return alternative_spellings
    # print(alternative_spellings)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("word",type=str)
    args = argparser.parse_args()
    results = get_syllable_based_misspellings(args.word)
    # print(results)
# 
