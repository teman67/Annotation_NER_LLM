import pandas as pd
from typing import List
import json
import streamlit as st


def format_tag_section(tag_df: pd.DataFrame) -> str:
    """
    Format tag definitions and examples from the CSV into a clear section.
    """
    section = ""
    for _, row in tag_df.iterrows():
        tag_name = row['tag_name']
        definition = row['definition']
        examples = row['examples']
        
        section += f"{tag_name}: {definition}\n"
        section += f"  Examples: {examples}\n\n"
    
    return section


def build_annotation_prompt(tag_df: pd.DataFrame, chunk_text: str,
                            few_shot_examples: list = None) -> str:
    """
    Build prompt with hard exclusion on tag label variants including plural/singular, separators, and casing.
    """
    tag_section = format_tag_section(tag_df)

    # The rest of the codes are removed to make it private.
        
        return variants

    # Generate exclusion terms from tag names
    exclusion_terms = set()
    if 'tag_name' in tag_df.columns:
        for tag_name in tag_df['tag_name']:
            exclusion_terms.update(generate_exclusion_variants(tag_name))
        
        # Create a formatted exclusion list
        exclusion_list = ", ".join(f'"{term}"' for term in sorted(exclusion_terms))
    else:
        exclusion_list = ""
    
    # Format few-shot examples if provided
    few_shot_section = ""
    if few_shot_examples:
        few_shot_section = "\n" + "="*50 + "\nFEW-SHOT EXAMPLES:\n" + "="*50 + "\n"
        for i, example in enumerate(few_shot_examples[:3], 1):
            few_shot_section += f"\nExample {i}:\n"
            few_shot_section += f"Text: \"{example['text']}\"\n"
            few_shot_section += f"Output: {example['output']}\n"
            few_shot_section += "-" * 30 + "\n"

    prompt = f"""You are a scientific named entity recognition (NER) expert.  # The rest of the codes are removed to make it private."""

    return prompt



def build_evaluation_prompt(tag_df: pd.DataFrame, entities: list) -> str:
    """
    Build a prompt for evaluating whether annotated entities are correctly labeled according to tag definitions.
    """
    tag_section = format_tag_section(tag_df)

    # Format annotated entities
    entities_text = ""
    for i, entity in enumerate(entities):
        entities_text += f"Entity {i+1}:\n"
        entities_text += f"- Text: \"{entity['text']}\"\n"
        entities_text += f"- Assigned Label: {entity['label']}\n"
        entities_text += f"- Character Range: [{entity['start_char']}:{entity['end_char']}]\n\n"

    prompt = f"""
You are a domain expert in named entity recognition (NER) annotation quality control. Your task is to  # The rest of the codes are removed to make it private.
"""

    return prompt
