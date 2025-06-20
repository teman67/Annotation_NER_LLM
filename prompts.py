# prompts.py

import pandas as pd
from typing import List

def format_tag_section(tag_df: pd.DataFrame) -> str:
    """
    Generate a string with all tag_name, definition, and examples from the uploaded CSV.
    """
    tag_texts = []
    for _, row in tag_df.iterrows():
        tag_block = (
            f"TAG: {row['tag_name']}\n"
            f"Definition: {row['definition']}\n"
            f"Examples: {row['examples']}\n"
        )
        tag_texts.append(tag_block)
    return "\n".join(tag_texts)

def build_annotation_prompt(tag_df: pd.DataFrame, chunk_text: str) -> str:
    """
    Constructs the full prompt including tag definitions and the target text to annotate.
    """
    tag_section = format_tag_section(tag_df)

    prompt = f"""
You are a domain expert in scientific text annotation. Your task is to perform Named Entity Recognition (NER)
on the following text using the tag definitions and examples provided.

Tag definitions and examples:
-----------------------------
{tag_section}

Instructions:
- Annotate only the text spans that clearly match the definitions.
- For each recognized entity, return a JSON object with:
  - start_char: character index where the entity starts
  - end_char: character index where it ends
  - text: the exact span
  - label: the matching tag

Return your output as a JSON list like this:
[
  {{"start_char": 12, "end_char": 25, "text": "graphene oxide", "label": "MATERIAL"}},
  {{"start_char": 56, "end_char": 72, "text": "X-ray diffraction", "label": "METHOD"}}
]

Text to annotate:
-----------------
{chunk_text}
""".strip()

    return prompt
