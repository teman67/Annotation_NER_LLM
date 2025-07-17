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

    prompt = f"""You are a scientific named entity recognition (NER) expert. Extract entities that match the SEMANTIC MEANING of tag definitions, not the literal tag labels themselves.

STRICT RULES:
• STRICTLY DO NOT annotate any of the following tag names, even if they appear in a different form: {exclusion_list}
• These terms are considered category labels, NOT scientific entities.
• Variants include different capitalizations, plural/singular forms, and character changes like "-", "_", or space.
• Do not annotate these terms even if they appear as-is in the input.

GOOD ANNOTATIONS:
• Concrete examples of the category (e.g., "finite element analysis" for METHOD, "steel" for MATERIAL_TYPE)
• Specific, contextually grounded terms (e.g., "epoxy resin", "finite element simulation", "heat-treated steel")
• These should clearly belong to one of the tag definitions.

BAD ANNOTATIONS:
• Abstract category names (e.g., "method", "process", "material properties")
• Any term that appears in the exclusion list above
• Generic label terms (e.g., "type", "method", "result", "condition")
• Anything from the exclusion list — even if it appears verbatim in the text.


TAG DEFINITIONS AND EXAMPLES:

{tag_section} {few_shot_section}

TARGET TEXT TO ANNOTATE:

{chunk_text}


OUTPUT FORMAT:

Return valid JSON array of entities with start_char, end_char, text, and label fields:"""

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
You are a domain expert in named entity recognition (NER) annotation quality control. Your task is to evaluate whether each annotated entity below has been labeled appropriately, based on the provided label definitions.

====================
LABEL DEFINITIONS:
{tag_section}
====================

====================
ANNOTATED ENTITIES:
{entities_text}
====================

For each entity, assess the following:
1. Does the entity’s **semantic meaning** align with the definition of the assigned label?
2. If not, suggest a better label from the available tags or recommend removal if no label fits.
3. Should the entity be removed entirely if it doesn't fit any tag?

⚠️ **Evaluation Guidelines**:
- Use the definitions above to guide your decision.
- Base your judgment on meaning and scientific or domain-specific correctness, not just keyword similarity.
- Be precise. Only suggest labels that clearly fit. Avoid vague justifications.
• Mark for deletion if entity doesn't match any available tag definition

✅ **Response Format** (as a JSON array):
[
  {{
    "entity_index": 0,
    "current_text": "original entity text",
    "current_label": "assigned_label",
    "is_correct": true/false,
    "recommendation": "keep" | "change_label" | "delete",
    "suggested_label": "new_label" | null,
    "reasoning": "Concise justification (max 300 characters)"
  }},
  ...
]

🔁 Return one JSON object per entity, in the same order they appear above.
🧠 Your reasoning must be helpful and actionable for improving annotation quality.
"""

    return prompt
