# prompts.py

def build_annotation_prompt(tag_df, text_chunk):
    """
    Build prompt for flat (traditional) annotation.
    """
    tag_definitions = []
    for _, row in tag_df.iterrows():
        tag_name = row['tag_name']
        definition = row['definition']
        examples = row['examples']
        tag_definitions.append(f"- **{tag_name}**: {definition}\n  Examples: {examples}")
    
    tag_section = "\n".join(tag_definitions)
    
    prompt = f"""You are an expert scientific text annotator. Your task is to identify and extract entities from the given text according to the provided tag definitions.

## Tag Definitions:
{tag_section}

## Instructions:
1. Read the text carefully and identify all entities that match the provided tag definitions
2. For each entity, provide the exact character positions (start_char, end_char)
3. Extract the exact text span for each entity
4. Assign the most appropriate tag label
5. Provide a confidence score between 0.0 and 1.0 for each annotation

## Text to Annotate:
{text_chunk}

## Output Format:
Return your annotations as a JSON array with the following structure:
[
  {{
    "start_char": <integer>,
    "end_char": <integer>,
    "text": "<exact text span>",
    "label": "<tag_name>",
    "confidence": <float between 0.0 and 1.0>
  }}
]

Make sure to:
- Use exact character positions from the original text
- Include only entities that clearly match the tag definitions
- Provide realistic confidence scores based on how well the entity fits the definition
- Return valid JSON format only, no additional text

JSON Array:"""

    return prompt


def build_custom_prompt(tag_df, text_chunk, custom_instructions=""):
    """
    Build a custom prompt with additional user-specified instructions.
    """
    base_prompt = build_annotation_prompt(tag_df, text_chunk)
    
    if custom_instructions:
        # Insert custom instructions before the output format section
        sections = base_prompt.split("## Output Format:")
        enhanced_prompt = f"{sections[0]}\n## Additional Instructions:\n{custom_instructions}\n\n## Output Format:{sections[1]}"
        return enhanced_prompt
    
    return base_prompt

def build_few_shot_prompt(tag_df, text_chunk, examples=None):
    """
    Build a few-shot prompt with examples to improve annotation quality.
    """
    tag_definitions = []
    for _, row in tag_df.iterrows():
        tag_name = row['tag_name']
        definition = row['definition']
        examples = row['examples']
        tag_definitions.append(f"- **{tag_name}**: {definition}\n  Examples: {examples}")
    
    tag_section = "\n".join(tag_definitions)
    
    # Default examples if none provided
    if examples is None:
        examples = [
            {
                "text": "The p53 protein regulates cell cycle progression.",
                "annotations": [
                    {
                        "start_char": 4,
                        "end_char": 7,
                        "text": "p53",
                        "label": "PROTEIN_NAME",
                        "confidence": 0.95
                    },
                    {
                        "start_char": 4,
                        "end_char": 15,
                        "text": "p53 protein",
                        "label": "PROTEIN",
                        "confidence": 0.9
                    }
                ]
            }
        ]
    
    example_section = ""
    for i, example in enumerate(examples, 1):
        example_section += f"\n### Example {i}:\nText: \"{example['text']}\"\nAnnotations: {example['annotations']}\n"
    
    prompt = f"""You are an expert scientific text annotator. Your task is to identify and extract entities from the given text according to the provided tag definitions.

## Tag Definitions:
{tag_section}

## Examples:
{example_section}

## Instructions:
1. Read the text carefully and identify all entities that match the provided tag definitions
2. Follow the annotation style shown in the examples above
3. For each entity, provide the exact character positions (start_char, end_char)
4. Extract the exact text span for each entity
5. Assign the most appropriate tag label
6. Provide a confidence score between 0.0 and 1.0 for each annotation

## Text to Annotate:
{text_chunk}

## Output Format:
Return your annotations as a JSON array with the following structure:
[
  {{
    "start_char": <integer>,
    "end_char": <integer>,
    "text": "<exact text span>",
    "label": "<tag_name>",
    "confidence": <float between 0.0 and 1.0>
  }}
]

Make sure to:
- Use exact character positions from the original text
- Include only entities that clearly match the tag definitions
- Provide realistic confidence scores based on how well the entity fits the definition
- Return valid JSON format only, no additional text

JSON Array:"""

    return prompt