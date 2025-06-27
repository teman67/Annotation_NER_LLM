# prompts.py

import pandas as pd
from typing import List
import json
import streamlit as st


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

# def build_annotation_prompt(tag_df: pd.DataFrame, chunk_text: str) -> str:
#     """
#     Constructs the full prompt including tag definitions and the target text to annotate.
#     """
#     tag_section = format_tag_section(tag_df)

#     prompt = f"""
# You are a domain expert in scientific text annotation. Your task is to perform Named Entity Recognition (NER)
# on the following text using the tag definitions and examples provided.

# Tag definitions and examples:
# -----------------------------
# {tag_section}

# Instructions:
# - Annotate only the text spans that clearly match the definitions.
# - For each recognized entity, return a JSON object with:
#   - start_char: character index where the entity starts
#   - end_char: character index where it ends
#   - text: the exact span
#   - label: the matching tag

# Return your output as a JSON list like this:
# [
#   {{"start_char": 12, "end_char": 25, "text": "graphene oxide", "label": "MATERIAL", "confidence": 0.85}},
#   {{"start_char": 56, "end_char": 72, "text": "X-ray diffraction", "label": "METHOD", "confidence": 0.95}}
# ]

# Text to annotate:
# -----------------
# {chunk_text}
# """.strip()

#     return prompt



# ############

# def build_annotation_prompt(tag_df: pd.DataFrame, chunk_text: str) -> str:
#     """
#     Constructs an optimized prompt for scientific NER with explicit exclusion rules
#     and improved instruction clarity.
#     """
#     tag_section = format_tag_section(tag_df)
    
#     # Extract tag names for exclusion list
#     tag_names = tag_df['tag_name'].str.lower().tolist() if 'tag_name' in tag_df.columns else []
#     exclusion_examples = ", ".join([f'"{name}"' for name in tag_names[:5]])  # Show first 5 as examples

#     prompt = f"""You are a scientific text annotation expert performing Named Entity Recognition (NER).

# ANNOTATION RULES:
# 1. Annotate text spans that match the semantic meaning of tag definitions below
# 2. Do NOT annotate words that are identical to tag names themselves
# 3. Focus on substantive scientific entities, not generic descriptive words
# 4. Ensure annotations capture complete entity boundaries (full phrases, not fragments)

# EXCLUSION RULE - Do NOT annotate these exact words when they appear as standalone terms:
# {exclusion_examples}{"..." if len(tag_names) > 5 else ""}

# TAG DEFINITIONS:
# {tag_section}

# OUTPUT FORMAT:
# Return a JSON array of entities. Each entity must include:
# - start_char: starting character index (0-based)
# - end_char: ending character index (exclusive)
# - text: exact text span
# - label: matching tag name

# Example output:
# [
#   {{"start_char": 12, "end_char": 25, "text": "graphene oxide", "label": "MATERIAL"}},
#   {{"start_char": 56, "end_char": 72, "text": "X-ray diffraction", "label": "METHOD"}}
# ]

# VALIDATION CHECKLIST:
# - ✓ Entity text matches definition semantically
# - ✓ Boundaries capture complete phrases
# - ✓ Not annotating tag names themselves
# - ✓ JSON format is valid

# TEXT TO ANNOTATE:
# {chunk_text}

# JSON OUTPUT:""".strip()

#     return prompt


import pandas as pd

def build_annotation_prompt(tag_df: pd.DataFrame, chunk_text: str,
                            few_shot_examples: list = None) -> str:
    """
    Build prompt with hard exclusion on tag label variants including plural/singular, separators, and casing.
    """
    tag_section = format_tag_section(tag_df)

    def generate_exclusion_variants(name: str) -> set:
        suffix_map = {
            "properties": "property", "property": "properties",
            "methods": "method", "method": "methods",
            "types": "type", "type": "types",
            "conditions": "condition", "condition": "conditions",
            "processes": "process", "process": "processes",
            "analyses": "analysis", "analysis": "analyses",
            "results": "result", "result": "results"
        }

        base = name.lower().replace('_', ' ').replace('-', ' ')
        tokens = base.split()
        variants = set()
        separators = [' ', '-', '_']
        casings = [str.lower, str.upper, str.title, str.capitalize]

        for sep in separators:
            combined = sep.join(tokens)
            for case_fn in casings:
                form = case_fn(combined)
                variants.add(form)

                # Add plural/singular variants
                for plural, singular in suffix_map.items():
                    if form.endswith(plural):
                        variants.add(case_fn(form[:-len(plural)] + singular))
                    if form.endswith(singular):
                        variants.add(case_fn(form[:-len(singular)] + plural))

        return variants

    exclusion_terms = set()
    if 'tag_name' in tag_df.columns:
        for tag_name in tag_df['tag_name']:
            exclusion_terms.update(generate_exclusion_variants(tag_name))
        exclusion_list = ", ".join(f'"{term}"' for term in sorted(exclusion_terms))
    else:
        exclusion_list = ""

    few_shot_section = ""
    if few_shot_examples:
        few_shot_section = "\nFEW-SHOT EXAMPLES:\n"
        for i, example in enumerate(few_shot_examples[:3], 1):
            few_shot_section += f"\nExample {i}:\nText: \"{example['text']}\"\nOutput: {example['output']}\n"
        few_shot_section += "\n"

    prompt = f"""You are a scientific NER annotation expert. Extract entities that match the SEMANTIC MEANING of tag definitions, not the literal tag labels themselves.

STRICT RULES:
• STRICTLY DO NOT annotate any of the following tag names, even if they appear in a different form: {exclusion_list}
• These terms are considered category labels, NOT scientific entities.
• Variants include different capitalizations, plural/singular forms, and character changes like "-", "_", or space.
• Do not annotate these terms even if they appear as-is in the input.

GOOD ANNOTATIONS:
• Concrete examples of the category (e.g., "finite element analysis" for METHOD, "steel" for MATERIAL_TYPE)

BAD ANNOTATIONS:
• Abstract category names (e.g., "method", "process", "material properties")
• Any term that appears in the exclusion list above

TAG DEFINITIONS:
{tag_section}{few_shot_section}
TARGET TEXT:
{chunk_text}

Return valid JSON array of entities with start_char, end_char, text, and label fields:"""

    return prompt



# def build_annotation_prompt_contextual(tag_df: pd.DataFrame, chunk_text: str, 
#                                      context_window: str = None) -> str:
#     """
#     Version that includes surrounding context for better boundary detection.
#     """
#     tag_section = format_tag_section(tag_df)
#     tag_names_lower = set(tag_df['tag_name'].str.lower()) if 'tag_name' in tag_df.columns else set()
    
#     context_section = ""
#     if context_window:
#         context_section = f"\nSURROUNDING CONTEXT (for reference only, do not annotate):\n{context_window}\n"

#     prompt = f"""Scientific NER Task: Identify entities matching the provided tag definitions.

# ANNOTATION GUIDELINES:
# 1. Match entities by semantic meaning, not surface form
# 2. Exclude tag names when used as generic descriptors: {list(tag_names_lower)}
# 3. Include complete phrases (e.g., "transmission electron microscopy")
# 4. Verify entity boundaries align with natural language units
# 5. Only annotate spans you're confident about

# {tag_section}{context_section}

# CURRENT TEXT CHUNK:
# {chunk_text}

# Output format - JSON array:
# [{{"start_char": int, "end_char": int, "text": "span", "label": "TAG", "confidence": "float"}}]

# Annotations:""".strip()

#     return prompt



# Add this function to prompts_flat.py

def build_evaluation_prompt(tag_df: pd.DataFrame, entities: list) -> str:
    """
    Build prompt for evaluating whether annotated entities match their tag definitions.
    """
    tag_section = format_tag_section(tag_df)
    
    # Format entities for evaluation
    entities_text = ""
    for i, entity in enumerate(entities):
        entities_text += f"Entity {i+1}:\n"
        entities_text += f"- Text: \"{entity['text']}\"\n"
        entities_text += f"- Current Label: {entity['label']}\n"
        entities_text += f"- Position: [{entity['start_char']}:{entity['end_char']}]\n\n"
    
    prompt = f"""You are an expert annotation evaluator. Your task is to evaluate whether each annotated entity matches its assigned label definition.

TAG DEFINITIONS:
{tag_section}

ANNOTATED ENTITIES TO EVALUATE:
{entities_text}

For each entity, evaluate:
1. Does the entity text semantically match the definition of its assigned label?
2. If not, what would be a better label from the available tags (if any)?


EVALUATION RULES:
• Focus on semantic meaning, not just keyword matching
• Consider context and scientific accuracy
• Only suggest labels that truly fit the entity


Return a JSON array with evaluation results for each entity:
[
  {{
    "entity_index": 0,
    "current_text": "original entity text",
    "current_label": "current_label",
    "is_correct": true/false,
    "confidence": 0.95,
    "recommendation": "keep/change_label",
    "suggested_label": "new_label or null",
    "reasoning": "explanation of the evaluation"
  }}
]

Provide evaluation for ALL entities in the same order they were presented:"""

    return prompt


### Backup
# def build_evaluation_prompt(tag_df: pd.DataFrame, entities: list) -> str:
#     """
#     Build prompt for evaluating whether annotated entities match their tag definitions.
#     """
#     tag_section = format_tag_section(tag_df)
    
#     # Format entities for evaluation
#     entities_text = ""
#     for i, entity in enumerate(entities):
#         entities_text += f"Entity {i+1}:\n"
#         entities_text += f"- Text: \"{entity['text']}\"\n"
#         entities_text += f"- Current Label: {entity['label']}\n"
#         entities_text += f"- Position: [{entity['start_char']}:{entity['end_char']}]\n\n"
    
#     prompt = f"""You are an expert annotation evaluator. Your task is to evaluate whether each annotated entity matches its assigned label definition.

# TAG DEFINITIONS:
# {tag_section}

# ANNOTATED ENTITIES TO EVALUATE:
# {entities_text}

# For each entity, evaluate:
# 1. Does the entity text semantically match the definition of its assigned label?
# 2. If not, what would be a better label from the available tags (if any)?
# 3. Should the entity be removed entirely if it doesn't fit any tag?

# EVALUATION RULES:
# • Focus on semantic meaning, not just keyword matching
# • Consider context and scientific accuracy
# • Only suggest labels that truly fit the entity
# • Mark for deletion if entity doesn't match any available tag definition

# Return a JSON array with evaluation results for each entity:
# [
#   {{
#     "entity_index": 0,
#     "current_text": "original entity text",
#     "current_label": "current_label",
#     "is_correct": true/false,
#     "confidence": 0.95,
#     "recommendation": "keep/change_label/delete",
#     "suggested_label": "new_label or null",
#     "reasoning": "explanation of the evaluation"
#   }}
# ]

# Provide evaluation for ALL entities in the same order they were presented:"""

#     return prompt