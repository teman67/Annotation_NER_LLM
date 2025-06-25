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



def build_evaluation_prompt(tag_df, text, entities):
    """
    Build a prompt for the LLM to evaluate and optimize annotations.
    """
    # Get valid tag names from CSV
    valid_tags = tag_df['tag_name'].tolist()
    
    # Create tag definitions string
    tag_definitions = []
    for _, row in tag_df.iterrows():
        tag_definitions.append(f"- {row['tag_name']}: {row['definition']}")
        if pd.notna(row['examples']) and row['examples'].strip():
            tag_definitions.append(f"  Examples: {row['examples']}")
    
    tag_definitions_str = "\n".join(tag_definitions)
    
    # Create current annotations string
    current_annotations = []
    for i, entity in enumerate(entities):
        current_annotations.append(f"{i+1}. Text: '{entity['text']}' | Label: '{entity['label']}' | Position: [{entity['start_char']}:{entity['end_char']}]")
    
    current_annotations_str = "\n".join(current_annotations)
    
    prompt = f"""You are an expert annotation evaluator. Your task is to review and optimize the existing annotations based on the provided tag definitions.

VALID TAG DEFINITIONS:
{tag_definitions_str}

CURRENT ANNOTATIONS TO EVALUATE:
{current_annotations_str}

ORIGINAL TEXT EXCERPT (first 1000 characters):
{text[:1000]}...

INSTRUCTIONS:
1. Review each annotation and check if the label matches one of the valid tag names exactly
2. If a label doesn't match any valid tag, suggest the most appropriate valid tag based on the text and definitions
3. If an annotation seems incorrect or doesn't fit any valid tag, suggest removing it
4. Check if any annotations are duplicates or overlapping inappropriately
5. Evaluate if the text span is appropriate for the assigned label

OUTPUT FORMAT:
Return a JSON array with your recommendations. For each annotation, include:
- "annotation_id": The original annotation number (1-based)
- "original_text": The original text span
- "original_label": The original label
- "action": One of ["keep", "relabel", "remove"]
- "new_label": If action is "relabel", provide the correct valid tag name
- "confidence": A score from 0.0 to 1.0 indicating your confidence in this recommendation
- "reason": Brief explanation of your decision

Example output:
[
  {{
    "annotation_id": 1,
    "original_text": "diabetes",
    "original_label": "disease",
    "action": "relabel",
    "new_label": "Disease",
    "confidence": 0.9,
    "reason": "Label should be capitalized to match valid tag name 'Disease'"
  }},
  {{
    "annotation_id": 2,
    "original_text": "patient",
    "original_label": "InvalidTag",
    "action": "remove",
    "new_label": null,
    "confidence": 0.8,
    "reason": "InvalidTag is not in valid tag list and 'patient' doesn't fit any valid categories"
  }}
]

IMPORTANT: Only return the JSON array, no additional text or explanation."""
    
    return prompt

def parse_evaluation_response(response_text):
    """
    Parse the LLM evaluation response with improved error handling.
    """
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Try direct JSON parsing first
        recommendations = json.loads(response_text)
        
        if isinstance(recommendations, list):
            # Validate recommendation structure
            valid_recommendations = []
            required_fields = ["annotation_id", "original_text", "original_label", "action", "confidence", "reason"]
            
            for rec in recommendations:
                if isinstance(rec, dict) and all(field in rec for field in required_fields):
                    # Validate action values
                    if rec["action"] in ["keep", "relabel", "remove"]:
                        valid_recommendations.append(rec)
                    else:
                        st.warning(f"Invalid action '{rec['action']}' in recommendation: {rec}")
                else:
                    st.warning(f"Invalid recommendation structure: {rec}")
            
            return valid_recommendations
        else:
            st.error(f"Response is not a list: {type(recommendations)}")
            return []
            
    except json.JSONDecodeError as e:
        # Try to extract JSON array from text
        try:
            first_bracket = response_text.find('[')
            last_bracket = response_text.rfind(']')
            
            if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
                json_str = response_text[first_bracket:last_bracket+1]
                recommendations = json.loads(json_str)
                
                # Validate and return
                valid_recommendations = []
                required_fields = ["annotation_id", "original_text", "original_label", "action", "confidence", "reason"]
                
                for rec in recommendations:
                    if isinstance(rec, dict) and all(field in rec for field in required_fields):
                        if rec["action"] in ["keep", "relabel", "remove"]:
                            valid_recommendations.append(rec)
                
                return valid_recommendations
            else:
                raise ValueError("No valid JSON array found")
                
        except (json.JSONDecodeError, ValueError):
            st.error(f"Failed to parse evaluation response: {e}")
            st.error(f"Raw response preview: {response_text[:200]}...")
            return []

def apply_evaluation_recommendations(entities, recommendations):
    """
    Apply the LLM recommendations to the entities.
    Returns updated entities and application stats.
    """
    updated_entities = []
    stats = {
        'total_recommendations': len(recommendations),
        'kept': 0,
        'relabeled': 0,
        'removed': 0,
        'failed_to_apply': 0,
        'changes_made': []
    }
    
    # Create a mapping from annotation_id to entity index (1-based to 0-based)
    id_to_index = {i+1: i for i in range(len(entities))}
    
    # Track which entities to keep/modify
    entities_to_process = list(range(len(entities)))
    
    for rec in recommendations:
        annotation_id = rec['annotation_id']
        action = rec['action']
        
        # Convert 1-based ID to 0-based index
        entity_index = id_to_index.get(annotation_id)
        
        if entity_index is None or entity_index >= len(entities):
            stats['failed_to_apply'] += 1
            continue
        
        entity = entities[entity_index]
        
        if action == "keep":
            stats['kept'] += 1
            
        elif action == "relabel":
            new_label = rec.get('new_label')
            if new_label:
                old_label = entity['label']
                entity['label'] = new_label
                stats['relabeled'] += 1
                stats['changes_made'].append({
                    'type': 'relabel',
                    'text': entity['text'],
                    'old_label': old_label,
                    'new_label': new_label,
                    'reason': rec['reason']
                })
            else:
                stats['failed_to_apply'] += 1
                
        elif action == "remove":
            # Mark for removal
            if entity_index in entities_to_process:
                entities_to_process.remove(entity_index)
            stats['removed'] += 1
            stats['changes_made'].append({
                'type': 'remove',
                'text': entity['text'],
                'label': entity['label'],
                'reason': rec['reason']
            })
        
        else:
            stats['failed_to_apply'] += 1
    
    # Create updated entities list (excluding removed ones)
    updated_entities = [entities[i] for i in entities_to_process]
    
    return updated_entities, stats

def run_evaluation_agent(text, entities, tag_df, client, temperature=0.1, max_tokens=2000):
    """
    Run the evaluation agent to optimize annotations.
    """
    if not entities:
        st.warning("No entities to evaluate.")
        return entities, {}
    
    st.write(f"🤖 Evaluating {len(entities)} annotations with LLM agent...")
    
    with st.spinner("🧠 LLM is analyzing annotations..."):
        # Build evaluation prompt
        prompt = build_evaluation_prompt(tag_df, text, entities)
        
        # Get LLM response
        response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        
        # Parse recommendations
        recommendations = parse_evaluation_response(response)
        
        if not recommendations:
            st.error("Failed to get valid recommendations from LLM.")
            return entities, {}
        
        # Show recommendations to user
        st.write("### 🤖 LLM Evaluation Recommendations")
        
        # Create a dataframe for display
        rec_data = []
        for rec in recommendations:
            rec_data.append({
                "ID": rec['annotation_id'],
                "Text": rec['original_text'],
                "Original Label": rec['original_label'],
                "Action": rec['action'],
                "New Label": rec.get('new_label', ''),
                "Confidence": f"{rec['confidence']:.2f}",
                "Reason": rec['reason']
            })
        
        rec_df = pd.DataFrame(rec_data)
        st.dataframe(rec_df, use_container_width=True)
        
        # Ask user for confirmation
        st.write("### 📋 Review Recommendations")
        
        col1, col2 = st.columns(2)
        with col1:
            apply_all = st.button("✅ Apply All Recommendations", key="apply_all_recs")
        with col2:
            apply_selected = st.button("🔍 Apply Selected Only", key="apply_selected_recs")
        
        if apply_all:
            # Apply all recommendations
            updated_entities, stats = apply_evaluation_recommendations(entities, recommendations)
            return updated_entities, stats, recommendations
        elif apply_selected:
            # Let user select which recommendations to apply
            selected_ids = st.multiselect(
                "Select recommendation IDs to apply:",
                options=[rec['annotation_id'] for rec in recommendations],
                default=[rec['annotation_id'] for rec in recommendations if rec['action'] != 'remove'],
                key="selected_recommendation_ids"
            )
            
            if st.button("Apply Selected Recommendations", key="confirm_selected"):
                selected_recs = [rec for rec in recommendations if rec['annotation_id'] in selected_ids]
                updated_entities, stats = apply_evaluation_recommendations(entities, selected_recs)
                return updated_entities, stats, selected_recs
        
        # Return original entities if no action taken
        return entities, {}, recommendations