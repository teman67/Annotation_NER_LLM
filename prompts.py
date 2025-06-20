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



############

def build_annotation_prompt(tag_df: pd.DataFrame, chunk_text: str) -> str:
    """
    Constructs an optimized prompt for scientific NER with explicit exclusion rules
    and improved instruction clarity.
    """
    tag_section = format_tag_section(tag_df)
    
    # Extract tag names for exclusion list
    tag_names = tag_df['tag_name'].str.lower().tolist() if 'tag_name' in tag_df.columns else []
    exclusion_examples = ", ".join([f'"{name}"' for name in tag_names[:5]])  # Show first 5 as examples

    prompt = f"""You are a scientific text annotation expert performing Named Entity Recognition (NER).

ANNOTATION RULES:
1. Annotate text spans that match the semantic meaning of tag definitions below
2. Do NOT annotate words that are identical to tag names themselves
3. Focus on substantive scientific entities, not generic descriptive words
4. Ensure annotations capture complete entity boundaries (full phrases, not fragments)

EXCLUSION RULE - Do NOT annotate these exact words when they appear as standalone terms:
{exclusion_examples}{"..." if len(tag_names) > 5 else ""}

TAG DEFINITIONS:
{tag_section}

OUTPUT FORMAT:
Return a JSON array of entities. Each entity must include:
- start_char: starting character index (0-based)
- end_char: ending character index (exclusive)
- text: exact text span
- label: matching tag name

Example output:
[
  {{"start_char": 12, "end_char": 25, "text": "graphene oxide", "label": "MATERIAL"}},
  {{"start_char": 56, "end_char": 72, "text": "X-ray diffraction", "label": "METHOD"}}
]

VALIDATION CHECKLIST:
- ✓ Entity text matches definition semantically
- ✓ Boundaries capture complete phrases
- ✓ Not annotating tag names themselves
- ✓ JSON format is valid

TEXT TO ANNOTATE:
{chunk_text}

JSON OUTPUT:""".strip()

    return prompt


def build_annotation_prompt_with_examples(tag_df: pd.DataFrame, chunk_text: str, 
                                        few_shot_examples: list = None) -> str:
    """
    Enhanced version with few-shot examples for better performance.
    """
    tag_section = format_tag_section(tag_df)
    tag_names = tag_df['tag_name'].str.lower().tolist() if 'tag_name' in tag_df.columns else []
    exclusion_list = ", ".join([f'"{name}"' for name in tag_names])
    
    few_shot_section = ""
    if few_shot_examples:
        few_shot_section = "\nFEW-SHOT EXAMPLES:\n"
        for i, example in enumerate(few_shot_examples[:3], 1):  # Limit to 3 examples
            few_shot_section += f"\nExample {i}:\nText: \"{example['text']}\"\nOutput: {example['output']}\n"
        few_shot_section += "\n"

    prompt = f"""You are a scientific NER annotation expert. Extract entities that match tag definitions while avoiding common pitfalls.

CRITICAL RULES:
• Annotate semantic matches to tag definitions, not literal tag name occurrences
• Skip these exact words when standalone: {exclusion_list}
• Capture complete entity spans (e.g., "scanning electron microscopy" not just "electron")
• Prioritize precision over recall - only annotate clear matches

TAG DEFINITIONS:
{tag_section}{few_shot_section}

TARGET TEXT:
{chunk_text}

Return valid JSON array of entities with start_char, end_char, text, and label fields:""".strip()

    return prompt


def build_annotation_prompt_contextual(tag_df: pd.DataFrame, chunk_text: str, 
                                     context_window: str = None) -> str:
    """
    Version that includes surrounding context for better boundary detection.
    """
    tag_section = format_tag_section(tag_df)
    tag_names_lower = set(tag_df['tag_name'].str.lower()) if 'tag_name' in tag_df.columns else set()
    
    context_section = ""
    if context_window:
        context_section = f"\nSURROUNDING CONTEXT (for reference only, do not annotate):\n{context_window}\n"

    prompt = f"""Scientific NER Task: Identify entities matching the provided tag definitions.

ANNOTATION GUIDELINES:
1. Match entities by semantic meaning, not surface form
2. Exclude tag names when used as generic descriptors: {list(tag_names_lower)}
3. Include complete phrases (e.g., "transmission electron microscopy")
4. Verify entity boundaries align with natural language units
5. Only annotate spans you're confident about

{tag_section}{context_section}

CURRENT TEXT CHUNK:
{chunk_text}

Output format - JSON array:
[{{"start_char": int, "end_char": int, "text": "span", "label": "TAG"}}]

Annotations:""".strip()

    return prompt