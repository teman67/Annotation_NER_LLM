import streamlit as st
import pandas as pd
import io
import json
import streamlit as st
import pandas as pd
from prompts_flat import build_annotation_prompt
from llm_clients import LLMClient
import html
import time
import streamlit.components.v1 as components
import colorsys
import hashlib
from typing import Dict, Tuple

def generate_label_colors(tag_list):
    """
    Generate visually distinct colors for each tag using hashing and HSL spacing.
    """
    label_colors = {}
    num_tags = len(tag_list)

    for i, tag in enumerate(sorted(tag_list)):
        # Generate hue spaced around the color wheel
        hue = i / num_tags
        lightness = 0.7
        saturation = 0.6
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert to hex
        color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        label_colors[tag] = color
    return label_colors

def get_model_pricing() -> Dict[str, Dict[str, float]]:
    """
    Get current model pricing information.
    Returns pricing per 1M tokens (input/output).
    """
    pricing = {
        # OpenAI models (per 1M tokens)
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        
        # Claude models (per 1M tokens)
        "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    }
    return pricing

def estimate_tokens(text: str, chunk_size: int = 1000) -> Dict[str, int]:
    """
    Estimate token counts for input text.
    Uses rough approximation: 1 token ≈ 4 characters for English text.
    """
    text_length = len(text)
    
    # Rough token estimation (4 characters per token)
    total_input_tokens = text_length // 4
    
    # Calculate number of chunks
    num_chunks = (text_length + chunk_size - 1) // chunk_size  # Ceiling division
    
    # Add tokens for system prompt and tag definitions (estimate)
    system_prompt_tokens = 500  # Rough estimate for system prompt
    tag_tokens_per_chunk = 200  # Rough estimate for tag definitions per chunk
    
    # Total input tokens includes text + system prompt + tag definitions
    total_input_tokens_with_overhead = total_input_tokens + (system_prompt_tokens * num_chunks) + (tag_tokens_per_chunk * num_chunks)
    
    return {
        "text_tokens": total_input_tokens,
        "system_overhead_tokens": (system_prompt_tokens * num_chunks) + (tag_tokens_per_chunk * num_chunks),
        "total_input_tokens": total_input_tokens_with_overhead,
        "num_chunks": num_chunks
    }

def estimate_output_tokens(num_chunks: int, max_tokens: int, num_tags: int) -> int:
    """
    Estimate output tokens based on chunks and expected annotations.
    """
    # Conservative estimate: assume we'll use most of max_tokens per chunk
    # but account for the fact that not all chunks will be fully utilized
    utilization_factor = 0.7  # Assume 70% utilization of max_tokens
    
    estimated_output_tokens = int(num_chunks * max_tokens * utilization_factor)
    
    return estimated_output_tokens

def calculate_cost_estimate(
    text: str,
    chunk_size: int,
    max_tokens: int,
    model: str,
    num_tags: int = 20
) -> Dict[str, float]:
    """
    Calculate cost estimate for annotation task.
    """
    pricing = get_model_pricing()
    
    if model not in pricing:
        return {"error": f"Pricing not available for model: {model}"}
    
    model_prices = pricing[model]
    
    # Get token estimates
    token_estimates = estimate_tokens(text, chunk_size)
    output_tokens = estimate_output_tokens(token_estimates["num_chunks"], max_tokens, num_tags)
    
    # Calculate costs (pricing is per 1M tokens)
    input_cost = (token_estimates["total_input_tokens"] / 1_000_000) * model_prices["input"]
    output_cost = (output_tokens / 1_000_000) * model_prices["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": token_estimates["total_input_tokens"],
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "num_chunks": token_estimates["num_chunks"],
        "text_tokens": token_estimates["text_tokens"],
        "system_overhead_tokens": token_estimates["system_overhead_tokens"]
    }

def display_cost_estimate(
    text: str,
    chunk_size: int,
    max_tokens: int,
    model: str,
    num_tags: int = 20
) -> None:
    """
    Display cost estimate in Streamlit.
    """
    if not text or len(text.strip()) == 0:
        st.info("📊 Upload text to see cost estimates")
        return
    
    cost_estimate = calculate_cost_estimate(text, chunk_size, max_tokens, model, num_tags)
    
    if "error" in cost_estimate:
        st.error(f"❌ {cost_estimate['error']}")
        return
    
    # Display cost estimate
    st.markdown("### 💰 Cost Estimate")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Cost",
            f"${cost_estimate['total_cost']:.4f}",
            help="Estimated total cost for this annotation task"
        )
    
    with col2:
        st.metric(
            "Input Cost",
            f"${cost_estimate['input_cost']:.4f}",
            help="Cost for processing input text and prompts"
        )
    
    with col3:
        st.metric(
            "Output Cost", 
            f"${cost_estimate['output_cost']:.4f}",
            help="Cost for generating annotations"
        )
    
    
    
    # Detailed breakdown
    with st.expander("📊 Detailed Breakdown", expanded=False):
        breakdown_data = {
            "Component": [
                "Text Tokens",
                "System Overhead",
                "Total Input Tokens",
                "Estimated Output Tokens",
                "Number of Chunks"
            ],
            "Count": [
                f"{cost_estimate['text_tokens']:,}",
                f"{cost_estimate['system_overhead_tokens']:,}",
                f"{cost_estimate['input_tokens']:,}",
                f"{cost_estimate['output_tokens']:,}",
                f"{cost_estimate['num_chunks']:,}"
            ],
            "Cost": [
                f"${cost_estimate['text_tokens'] / 1_000_000 * get_model_pricing()[model]['input']:.4f}",
                f"${cost_estimate['system_overhead_tokens'] / 1_000_000 * get_model_pricing()[model]['input']:.4f}",
                f"${cost_estimate['input_cost']:.4f}",
                f"${cost_estimate['output_cost']:.4f}",
                "N/A"
            ]
        }
        
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)
    
    # Cost optimization suggestions
    if cost_estimate['total_cost'] > 0.10:  # If cost > $0.10
        with st.expander("💡 Cost Optimization Tips", expanded=False):
            suggestions = []
            
            if cost_estimate['num_chunks'] > 20:
                suggestions.append("🔸 **Increase chunk size** to reduce the number of chunks and system overhead")
            
            if cost_estimate['output_tokens'] > cost_estimate['input_tokens']:
                suggestions.append("🔸 **Reduce max tokens** if you're not using the full output capacity")
            
            if model in ["gpt-4", "gpt-4o"]:
                suggestions.append("🔸 **Consider using gpt-4o-mini** for cost savings (about 10-20x cheaper)")
            
            if model == "claude-3-7-sonnet-20250219":
                suggestions.append("🔸 **Consider using claude-3-5-haiku** for cost savings (about 4x cheaper)")
            
            suggestions.append("🔸 **Process smaller text sections** to reduce overall cost")
            
            for suggestion in suggestions:
                st.markdown(suggestion)
    
    # Warning for high costs
    if cost_estimate['total_cost'] > 1.00:
        st.warning(f"⚠️ **High cost estimate: ${cost_estimate['total_cost']:.5f}**. Consider optimizing parameters or processing smaller text sections.")
    elif cost_estimate['total_cost'] > 0.50:
        st.info(f"💡 **Moderate cost: ${cost_estimate['total_cost']:.5f}**. Review settings if needed.")
    else:
        st.success(f"✅ **Low cost: ${cost_estimate['total_cost']:.5f}**. Good to go!")

def create_cost_comparison_table(text: str, chunk_size: int, max_tokens: int, num_tags: int = 20) -> pd.DataFrame:
    """
    Create a comparison table of costs across different models.
    """
    if not text or len(text.strip()) == 0:
        return pd.DataFrame()
    
    pricing = get_model_pricing()
    comparison_data = []
    
    for model_name in pricing.keys():
        cost_estimate = calculate_cost_estimate(text, chunk_size, max_tokens, model_name, num_tags)
        
        if "error" not in cost_estimate:
            # Determine model category
            if "gpt" in model_name.lower():
                provider = "OpenAI"
            elif "claude" in model_name.lower():
                provider = "Anthropic"
            else:
                provider = "Unknown"
            
            comparison_data.append({
                "Provider": provider,
                "Model": model_name,
                "Total Cost": f"${cost_estimate['total_cost']:.4f}",
                "Input Cost": f"${cost_estimate['input_cost']:.4f}",
                "Output Cost": f"${cost_estimate['output_cost']:.4f}",
                "Total Cost (Raw)": cost_estimate['total_cost']  # For sorting
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        # Sort by total cost
        df = df.sort_values("Total Cost (Raw)")
        # Remove the raw cost column
        df = df.drop("Total Cost (Raw)", axis=1)
        return df
    
    return pd.DataFrame()

def display_model_comparison(text: str, chunk_size: int, max_tokens: int, num_tags: int = 20) -> None:
    """
    Display model cost comparison in Streamlit.
    """
    if not text or len(text.strip()) == 0:
        return
    
    st.markdown("### 🔍 Model Cost Comparison")
    
    comparison_df = create_cost_comparison_table(text, chunk_size, max_tokens, num_tags)
    
    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Highlight the cheapest option
        cheapest_model = comparison_df.iloc[0]["Model"]
        st.success(f"💰 **Cheapest option:** {cheapest_model} at {comparison_df.iloc[0]['Total Cost']}")
    else:
        st.error("Could not generate cost comparison")

def display_processing_summary(text, tag_df, chunk_size, temperature, max_tokens, model_provider, model):
    """
    Display a comprehensive summary of processing parameters
    """
    chunks = chunk_text(text, chunk_size)
    
    st.markdown("### 📊 Processing Summary")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    # The rest of the codes are removed to make it private.
    
    st.markdown("---")

def display_chunk_progress(current_chunk, total_chunks, chunk_text, start_time=None):
    """
    Display attractive progress information for current chunk processing
    """
    # Progress bar
    progress = current_chunk / total_chunks
    st.progress(progress)
    
    # Progress info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # The rest of the codes are removed to make it private.

# Dynamic token calculation based on chunk size
def get_token_recommendations(chunk_size):
    if chunk_size <= 500:
        return 200, 800, 300
    elif chunk_size <= 1000:
        return 300, 1200, 400
    elif chunk_size <= 2000:
        return 500, 1800, 1000
    elif chunk_size <= 3000:
        return 700, 2500, 1400
    else:
        return 1000, 3000, 1800
    
def chunk_text(text: str, chunk_size: int):
    """
    Splits text into chunks of approximately chunk_size characters.
    Tries to split on newline or space to avoid cutting words abruptly.
    """
    chunks = []
    start = 0
    length = len(text)
    # The rest of the codes are removed to make it private.

def aggregate_entities(all_entities, offset):
    """
    Adjust entity character positions by offset (chunk start position in full text).
    """
    for ent in all_entities:
        ent['start_char'] += offset
        ent['end_char'] += offset
    return all_entities

def validate_entities_against_text(entities, full_text):
    """
    Final validation to ensure all entities actually exist in the original text.
    """
    validated_entities = []
    text_length = len(full_text)
    
    # The rest of the codes are removed to make it private.
    
    return validated_entities

def highlight_text_with_entities(text: str, entities: list, label_colors: dict) -> str:
    import html
    used_positions = set()
    highlighted = []
    last_pos = 0

    sorted_entities = sorted(entities, key=lambda x: x.get("start_char", 0))

    # The rest of the codes are removed to make it private.

    return ''.join(highlighted)

def display_annotated_entities():
    """
    Display annotated entities with highlighting and tooltips if they exist in session state.
    
    This function checks for annotated entities in Streamlit session state and renders
    them as an interactive HTML component with hover tooltips showing entity labels.
    """
    # The rest of the codes are removed to make it private.
       
        # Use Streamlit's HTML component to render the complete HTML
        components.html(full_html, height=400, scrolling=True)

def validate_annotations_streamlit(text, entities):
    """
    Validate that start_char and end_char positions in annotations match the actual text.
    Modified for Streamlit integration.
    
    Args:
        text (str): The source text
        entities (list): List of entity dictionaries
    
    Returns:
        dict: Validation results with errors and statistics
    """
    
    validation_results = {
        'total_entities': len(entities),
        'correct_entities': 0,
        'errors': [],
        'warnings': []
    }
    
    st.write(f"🔍 Validating {len(entities)} annotations...")
    
    # The rest of the codes are removed to make it private.
    
    return validation_results

def find_all_occurrences(text, pattern):
    """Find all occurrences of pattern in text"""
    positions = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        positions.append((pos, pos + len(pattern)))
        start = pos + 1
    return positions

def try_fuzzy_fix(text, expected_text, original_start, original_end):
    """Try to fix common annotation errors"""
    # Try removing/adding whitespace
    variations = [
        expected_text.strip(),
        expected_text.lstrip(),
        expected_text.rstrip(),
        ' ' + expected_text,
        expected_text + ' ',
        ' ' + expected_text + ' '
    ]
    
    # The rest of the codes are removed to make it private.
    
    return None

def fix_annotation_positions_streamlit(text, entities, strategy='closest'):
    """
    Automatically fix annotation positions by searching for the text.
    Modified for Streamlit integration.
    
    Args:
        text (str): The source text
        entities (list): List of entity dictionaries
        strategy (str): Strategy for handling multiple matches ('closest', 'first')
    
    Returns:
        tuple: (fixed_entities, stats)
    """
    
    fixed_entities = []
    stats = {
        'total': len(entities),
        'already_correct': 0,
        'fixed': 0,
        'unfixable': 0,
        'multiple_matches': 0
    }
    
    st.write(f"🔧 Attempting to fix {len(entities)} annotations...")
    
    # Create progress bar for fixing
    fix_progress = st.progress(0)
    fix_status = st.empty()
    
    # The rest of the codes are removed to make it private.
    
    # Clear progress indicators
    fix_progress.empty()
    fix_status.empty()
    
    return fixed_entities, stats


def evaluate_annotations_with_llm(entities, tag_df, client, temperature=0.1, max_tokens=2000):
    """
    Use LLM to evaluate whether annotations match their label definitions.
    FIXED VERSION with better error handling and entity tracking.
    """
    if not entities:
        st.warning("No entities to evaluate")
        return []
        
    from prompts_flat import build_evaluation_prompt
    
    # Split entities into batches if too many (to avoid token limits)
    batch_size = 20  # REDUCED from 50 to ensure better processing
    all_evaluations = []
    
    # The rest of the codes are removed to make it private.
    
    st.success(f"🎉 Evaluation completed! Generated {len(all_evaluations)} evaluations for {len(entities)} entities.")
    return all_evaluations


def parse_evaluation_response(response_text: str, batch_idx: int = None) -> list:
    """
    Parse the evaluation JSON response from LLM.
    ENHANCED VERSION with better error handling and recovery.
    """
    if not response_text or response_text.strip() == "":
        st.warning(f"⚠️ Empty evaluation response from LLM for batch {batch_idx if batch_idx is not None else 'unknown'}")
        return []
    
    response_text = response_text.strip()
    
    # The rest of the codes are removed to make it private.
    
    return []


def validate_evaluation_structure(evaluations: list) -> list:
    """
    Validate and clean evaluation results structure.
    """
    valid_evaluations = []
    required_fields = ["entity_index", "current_text", "current_label", "is_correct", "recommendation"]
    
    # The rest of the codes are removed to make it private.
    
    return valid_evaluations


def is_valid_evaluation_object(obj: dict) -> bool:
    """
    Check if an object has the minimum required fields for an evaluation.
    """
    required_fields = ["entity_index", "current_text", "current_label", "is_correct", "recommendation"]
    return all(field in obj for field in required_fields)

def clear_all_previous_data():
    """Clear all previous annotation and evaluation data when starting new annotation."""
    # Clear annotation data
    st.session_state.annotated_entities = []
    st.session_state.annotation_complete = False
    if 'editable_entities_df' in st.session_state:
        del st.session_state.editable_entities_df

    # Clear validation and fix results
    if 'validation_results' in st.session_state:
        del st.session_state.validation_results
    if 'fix_results' in st.session_state:
        del st.session_state.fix_results
    
    # Clear evaluation data (NEW)
    st.session_state.evaluation_results = []
    st.session_state.evaluation_complete = False
    st.session_state.evaluation_summary = {}

def apply_evaluation_recommendations(entities, evaluations, selected_indices):
    """
    Apply selected evaluation recommendations to entities.
    Returns updated entities and list of changes made.
    """
    if not entities:
        return [], ["No entities to process"]
   
    if not evaluations:
        return entities, ["No evaluations available"]
   
    updated_entities = entities.copy()
    changes_made = []
    entities_to_delete = []  # Use list to maintain order
   
    # Process all recommendations first (for label changes and mark deletions)
    # The rest of the codes are removed to make it private.
   
    return updated_entities, changes_made


def parse_llm_response(response_text: str, chunk_index: int = None):
    """
    Parse the JSON returned by LLM with improved error handling.
    Returns list of entities or empty list on error.
    """
    # Log the raw response for debugging
    if chunk_index is not None:
        st.write(f"**Debug - Chunk {chunk_index} Raw Response:**")
        with st.expander(f"Raw Response Content (Chunk {chunk_index})", expanded=False):
            st.text(repr(response_text))  # Use repr to show exact content including whitespace
    
    # Check if response is empty or None
    if not response_text or response_text.strip() == "":
        st.warning(f"⚠️ Empty response from LLM for chunk {chunk_index if chunk_index else 'unknown'}")
        return []
    
    # Clean the response text
    response_text = response_text.strip()
    
    # The rest of the codes are removed to make it private.
            return []



def run_annotation_pipeline(text, tag_df, client, temperature, max_tokens, chunk_size):
    """
    1. Chunk the text
    2. For each chunk, generate prompt and call LLM
    3. Parse and adjust entities with offset
    4. Aggregate and return full list of entities
    """
    chunks = chunk_text(text, chunk_size)
    all_entities = []
    char_pos = 0
    
    # Create a container for progress updates
    progress_container = st.container()
    
    start_time = time.time()
    
    # The rest of the codes are removed to make it private.
    
    return all_entities