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

def estimate_tokens(text):
    """
    Rough token estimation (1 token ≈ 4 characters for English text)
    """
    return len(text) // 4

def display_processing_summary(text, tag_df, chunk_size, temperature, max_tokens, model_provider, model):
    """
    Display a comprehensive summary of processing parameters
    """
    chunks = chunk_text(text, chunk_size)
    
    st.markdown("### 📊 Processing Summary")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Text Length", f"{len(text):,} chars", help="Total number of characters in the input text")
        st.metric("Estimated Tokens", f"{estimate_tokens(text):,}", help="Approximate number of tokens (1 token ≈ 4 characters)")
    
    with col2:
        st.metric("Number of Chunks", len(chunks), help="Text will be split into this many chunks")
        st.metric("Chunk Size", f"{chunk_size:,} chars", help="Maximum characters per chunk")
    
    with col3:
        st.metric("Total Tags", len(tag_df), help="Number of annotation tags available")
        st.metric("Temperature", temperature, help="LLM creativity setting (0=deterministic, 1=creative)")
    
    with col4:
        st.metric("Max Tokens/Response", max_tokens, help="Maximum tokens the LLM can generate per chunk")
        st.metric("Model", f"{model_provider}: {model}", help="Selected language model")
    
    # Display chunk information in an expandable section
    with st.expander("📋 Chunk Details", expanded=False):
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "Chunk #": i + 1,
                "Characters": len(chunk),
                "Est. Tokens": estimate_tokens(chunk),
                "Preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
        
        chunk_df = pd.DataFrame(chunk_data)
        st.dataframe(chunk_df, use_container_width=True)
    
    # Display tag information
    # with st.expander("🏷️ Tag Configuration", expanded=False):
    #     st.dataframe(tag_df[['tag_name', 'definition']], use_container_width=True)
    
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
    
    with col1:
        st.markdown(f"**Processing Chunk {current_chunk}/{total_chunks}**")
        if start_time:
            elapsed = time.time() - start_time
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            st.caption(f"⏱️ Elapsed: {elapsed:.1f}s | Estimated remaining: {remaining:.1f}s")
    
    with col2:
        st.metric("Progress", f"{progress:.1%}")
    
    with col3:
        st.metric("Chunk Size", f"{len(chunk_text):,} chars")
    
    # Chunk preview
    with st.expander(f"📄 Chunk {current_chunk} Preview", expanded=False):
        st.text_area(
            "Content Preview:", 
            value=chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text,
            height=100,
            disabled=True,
            key=f"chunk_preview_{current_chunk}"
        )

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
    while start < length:
        end = start + chunk_size
        if end >= length:
            chunks.append(text[start:])
            break
        # Try to split on last newline before end
        split_pos = text.rfind('\n', start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = text.rfind(' ', start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = end  # fallback hard cut

        chunks.append(text[start:split_pos].strip())
        start = split_pos
    return chunks

def aggregate_entities(all_entities, offset):
    """
    Adjust entity character positions by offset (chunk start position in full text).
    """
    for ent in all_entities:
        ent['start_char'] += offset
        ent['end_char'] += offset
    return all_entities

def highlight_text_with_entities(text: str, entities: list, label_colors: dict) -> str:
    import html
    used_positions = set()
    highlighted = []
    last_pos = 0

    sorted_entities = sorted(entities, key=lambda x: x.get("start_char", 0))

    for ent in sorted_entities:
        span = ent["text"]
        label = ent["label"]
        color = label_colors.get(label, "#e0e0e0")  # fallback if missing

        search_start = last_pos
        found = False
        while search_start < len(text):
            idx = text.find(span, search_start)
            if idx == -1:
                break
            if any(i in used_positions for i in range(idx, idx + len(span))):
                search_start = idx + 1
                continue
            else:
                highlighted.append(html.escape(text[last_pos:idx]))
                # Improved HTML with better tooltip styling
                highlighted.append(
                    f'<span style="background-color: {color}; font-weight: bold; padding: 2px 4px; '
                    f'border-radius: 3px; cursor: help; display: inline-block; '
                    f'border: 1px solid {color};" '
                    f'data-tooltip="{html.escape(label)}">'
                    f'{html.escape(span)}</span>'
                )
                used_positions.update(range(idx, idx + len(span)))
                last_pos = idx + len(span)
                found = True
                break

        if not found:
            continue

    # Append any remaining text after all entities
    highlighted.append(html.escape(text[last_pos:]))

    return ''.join(highlighted)

def display_annotated_entities():
    """
    Display annotated entities with highlighting and tooltips if they exist in session state.
    
    This function checks for annotated entities in Streamlit session state and renders
    them as an interactive HTML component with hover tooltips showing entity labels.
    """
    if 'annotated_entities' in st.session_state and st.session_state.annotated_entities:
        highlighted_html = highlight_text_with_entities(
            st.session_state.text_data,
            st.session_state.annotated_entities,
            st.session_state.label_colors
        )
        
        # Create a complete HTML document with inline CSS and JavaScript
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .annotation-container {{
                    font-family: Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.7;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                    margin: 10px 0;
                }}
               
                .annotation-container span[data-tooltip] {{
                    position: relative;
                    cursor: help;
                    transition: all 0.2s ease;
                }}
               
                .annotation-container span[data-tooltip]:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                }}
               
                .tooltip {{
                    visibility: hidden;
                    position: absolute;
                    bottom: 125%;
                    left: 50%;
                    transform: translateX(-50%);
                    background-color: #333;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: normal;
                    white-space: nowrap;
                    z-index: 1000;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    opacity: 0;
                    transition: opacity 0.3s, visibility 0.3s;
                }}
               
                .tooltip::after {{
                    content: '';
                    position: absolute;
                    top: 100%;
                    left: 50%;
                    margin-left: -6px;
                    border-width: 6px;
                    border-style: solid;
                    border-color: #333 transparent transparent transparent;
                }}
               
                .annotation-container span[data-tooltip]:hover .tooltip {{
                    visibility: visible;
                    opacity: 1;
                }}
            </style>
        </head>
        <body>
            <div class="annotation-container">
                {highlighted_html.replace('data-tooltip="', 'data-tooltip="').replace('">', '"><span class="tooltip"></span>')}
            </div>
           
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const spans = document.querySelectorAll('span[data-tooltip]');
                    spans.forEach(span => {{
                        const tooltip = span.querySelector('.tooltip');
                        if (tooltip) {{
                            tooltip.textContent = span.getAttribute('data-tooltip');
                        }}
                    }});
                }});
            </script>
        </body>
        </html>
        """
       
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
    
    # Create a progress bar for validation
    validation_progress = st.progress(0)
    validation_status = st.empty()
    
    for i, entity in enumerate(entities):
        # Update progress
        validation_progress.progress((i + 1) / len(entities))
        validation_status.text(f"Validating entity {i+1}/{len(entities)}: '{entity.get('text', 'N/A')}'")
        
        start_char = entity.get('start_char')
        end_char = entity.get('end_char')
        expected_text = entity.get('text')
        
        # Skip entities with missing required fields
        if None in [start_char, end_char, expected_text]:
            error_info = {
                'entity_index': i,
                'expected_text': expected_text,
                'start_char': start_char,
                'end_char': end_char,
                'error': 'Missing required fields',
                'label': entity.get('label', 'Unknown')
            }
            validation_results['errors'].append(error_info)
            continue
        
        # Extract actual text from the source using the character positions
        try:
            actual_text = text[start_char:end_char]
            
            # Check if texts match exactly
            if actual_text == expected_text:
                validation_results['correct_entities'] += 1
            else:
                error_info = {
                    'entity_index': i,
                    'expected_text': expected_text,
                    'actual_text': actual_text,
                    'start_char': start_char,
                    'end_char': end_char,
                    'label': entity.get('label', 'Unknown')
                }
                validation_results['errors'].append(error_info)
                
        except IndexError:
            error_info = {
                'entity_index': i,
                'expected_text': expected_text,
                'start_char': start_char,
                'end_char': end_char,
                'error': 'Index out of range',
                'label': entity.get('label', 'Unknown')
            }
            validation_results['errors'].append(error_info)
    
    # Clear progress indicators
    validation_progress.empty()
    validation_status.empty()
    
    # Additional checks
    # Check for overlapping annotations
    sorted_entities = sorted([e for e in entities if all(k in e for k in ['start_char', 'end_char'])], 
                           key=lambda x: x['start_char'])
    
    for i in range(len(sorted_entities) - 1):
        current = sorted_entities[i]
        next_entity = sorted_entities[i + 1]
        
        if current['end_char'] > next_entity['start_char']:
            warning = {
                'type': 'overlap',
                'entity1': current,
                'entity2': next_entity
            }
            validation_results['warnings'].append(warning)
    
    # Check for zero-length annotations
    zero_length = [e for e in entities if e.get('start_char') == e.get('end_char')]
    if zero_length:
        validation_results['warnings'].extend(zero_length)
    
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
    
    for variation in variations:
        positions = find_all_occurrences(text, variation)
        if positions:
            # Return the closest match to original position
            closest = min(positions, key=lambda x: abs(x[0] - original_start))
            return closest
    
    # Try case variations
    case_variations = [
        expected_text.lower(),
        expected_text.upper(),
        expected_text.capitalize()
    ]
    
    for variation in case_variations:
        positions = find_all_occurrences(text, variation)
        if positions:
            closest = min(positions, key=lambda x: abs(x[0] - original_start))
            return closest
    
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
    
    for i, entity in enumerate(entities):
        # Update progress
        fix_progress.progress((i + 1) / len(entities))
        fix_status.text(f"Processing entity {i+1}/{len(entities)}: '{entity.get('text', 'N/A')}'")
        
        expected_text = entity.get('text')
        start_char = entity.get('start_char')
        end_char = entity.get('end_char')
        
        # Skip entities with missing required fields
        if None in [expected_text, start_char, end_char]:
            fixed_entities.append(entity)
            stats['unfixable'] += 1
            continue
        
        # Check if current position is correct
        try:
            if start_char >= 0 and end_char <= len(text) and text[start_char:end_char] == expected_text:
                fixed_entities.append(entity)
                stats['already_correct'] += 1
                continue
        except:
            pass
        
        # Try to find the text in the document
        found_positions = find_all_occurrences(text, expected_text)
        
        if not found_positions:
            # Try fuzzy matching for common issues
            fixed_pos = try_fuzzy_fix(text, expected_text, start_char, end_char)
            if fixed_pos:
                entity_copy = entity.copy()
                entity_copy['start_char'] = fixed_pos[0]
                entity_copy['end_char'] = fixed_pos[1]
                fixed_entities.append(entity_copy)
                stats['fixed'] += 1
            else:
                # Text not found, keep original but mark as unfixable
                fixed_entities.append(entity)
                stats['unfixable'] += 1
        elif len(found_positions) == 1:
            # Only one match found, use it
            new_start, new_end = found_positions[0]
            entity_copy = entity.copy()
            entity_copy['start_char'] = new_start
            entity_copy['end_char'] = new_end
            fixed_entities.append(entity_copy)
            stats['fixed'] += 1
        else:
            # Multiple matches found
            stats['multiple_matches'] += 1
            
            if strategy == 'closest':
                # Choose the closest to original position
                closest_pos = min(found_positions, key=lambda x: abs(x[0] - start_char))
                entity_copy = entity.copy()
                entity_copy['start_char'] = closest_pos[0]
                entity_copy['end_char'] = closest_pos[1]
                fixed_entities.append(entity_copy)
                stats['fixed'] += 1
            elif strategy == 'first':
                # Use the first occurrence
                first_pos = found_positions[0]
                entity_copy = entity.copy()
                entity_copy['start_char'] = first_pos[0]
                entity_copy['end_char'] = first_pos[1]
                fixed_entities.append(entity_copy)
                stats['fixed'] += 1
    
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
    
    entity_batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]
    
    st.write(f"📊 Processing {len(entities)} entities in {len(entity_batches)} batches...")
    
    for batch_idx, entity_batch in enumerate(entity_batches):
        st.write(f"🤖 Evaluating batch {batch_idx + 1}/{len(entity_batches)} ({len(entity_batch)} entities)...")
        
        try:
            prompt = build_evaluation_prompt(tag_df, entity_batch)
            
            with st.spinner(f"LLM evaluating batch {batch_idx + 1}..."):
                response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                
                # Debug: Show raw response
                with st.expander(f"🔍 Debug: Batch {batch_idx + 1} Raw Response", expanded=False):
                    st.text(f"Response length: {len(response) if response else 0} characters")
                    st.code(response[:500] + "..." if len(response) > 500 else response, language="text")
                
                batch_evaluations = parse_evaluation_response(response, batch_idx)
                
                # CRITICAL FIX: Ensure we have evaluation for each entity in batch
                if len(batch_evaluations) != len(entity_batch):
                    st.warning(f"⚠️ Batch {batch_idx + 1}: Expected {len(entity_batch)} evaluations, got {len(batch_evaluations)}")
                    
                    # Fill missing evaluations with default values
                    for i in range(len(entity_batch)):
                        global_entity_idx = batch_idx * batch_size + i
                        
                        # Check if we have evaluation for this entity
                        found_eval = False
                        for eval_result in batch_evaluations:
                            if eval_result.get('entity_index') == i:  # Local batch index
                                # Update to global index
                                eval_result['entity_index'] = global_entity_idx
                                found_eval = True
                                break
                        
                        # If no evaluation found, create a default one
                        if not found_eval:
                            entity = entity_batch[i]
                            default_eval = {
                                'entity_index': global_entity_idx,
                                'current_text': entity.get('text', ''),
                                'current_label': entity.get('label', ''),
                                'is_correct': False,  # Conservative default
                                'recommendation': 'manual_review',
                                'reasoning': 'LLM evaluation failed - requires manual review',
                                'suggested_label': entity.get('label', ''),
                                'confidence': 0.0
                            }
                            batch_evaluations.append(default_eval)
                            st.warning(f"Created default evaluation for entity {global_entity_idx}")
                else:
                    # Update entity indices to global indices
                    for i, eval_result in enumerate(batch_evaluations):
                        # eval_result['entity_index'] = batch_idx * batch_size + eval_result.get('entity_index', i)
                        eval_result['entity_index'] = batch_idx * batch_size + i
                
                all_evaluations.extend(batch_evaluations)
                st.success(f"✅ Batch {batch_idx + 1} completed! Processed {len(batch_evaluations)} evaluations.")
                
        except Exception as e:
            st.error(f"❌ Batch {batch_idx + 1} failed: {e}")
            
            # Create default evaluations for failed batch
            for i, entity in enumerate(entity_batch):
                global_entity_idx = batch_idx * batch_size + i
                default_eval = {
                    'entity_index': global_entity_idx,
                    'current_text': entity.get('text', ''),
                    'current_label': entity.get('label', ''),
                    'is_correct': False,
                    'recommendation': 'manual_review',
                    'reasoning': f'Batch evaluation failed: {str(e)}',
                    'suggested_label': entity.get('label', ''),
                    'confidence': 0.0
                }
                all_evaluations.append(default_eval)
            
            st.warning(f"Created {len(entity_batch)} default evaluations for failed batch")
    
    # FINAL VERIFICATION: Ensure we have evaluation for every entity
    if len(all_evaluations) != len(entities):
        st.error(f"❌ CRITICAL: Expected {len(entities)} evaluations, got {len(all_evaluations)}")
        
        # Find missing entities and create default evaluations
        evaluated_indices = set(eval_result.get('entity_index', -1) for eval_result in all_evaluations)
        missing_indices = set(range(len(entities))) - evaluated_indices
        
        if missing_indices:
            st.warning(f"Creating default evaluations for {len(missing_indices)} missing entities: {sorted(missing_indices)}")
            
            for entity_idx in missing_indices:
                if entity_idx < len(entities):
                    entity = entities[entity_idx]
                    default_eval = {
                        'entity_index': entity_idx,
                        'current_text': entity.get('text', ''),
                        'current_label': entity.get('label', ''),
                        'is_correct': False,
                        'recommendation': 'manual_review',
                        'reasoning': 'Missing from LLM evaluation - requires manual review',
                        'suggested_label': entity.get('label', ''),
                        'confidence': 0.0
                    }
                    all_evaluations.append(default_eval)
        
        # Sort evaluations by entity_index to maintain order
        all_evaluations.sort(key=lambda x: x.get('entity_index', 0))
    
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
    
    try:
        # Method 1: Try direct JSON parsing first
        evaluations = json.loads(response_text)
        if isinstance(evaluations, list):
            valid_evaluations = validate_evaluation_structure(evaluations)
            if valid_evaluations:
                return valid_evaluations
        else:
            st.warning(f"Evaluation response is not a list: {type(evaluations)}")
            
    except json.JSONDecodeError:
        pass
    
    # Method 2: Try to extract JSON from markdown code blocks
    try:
        import re
        
        # Look for JSON content between ```json and ``` or ``` and ```
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```',
            r'(\[(?:[^[\]]*|\[[^[\]]*\])*\])'  # Find any JSON array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    evaluations = json.loads(match.strip())
                    if isinstance(evaluations, list):
                        valid_evaluations = validate_evaluation_structure(evaluations)
                        if valid_evaluations:
                            st.info(f"Recovered {len(valid_evaluations)} evaluations using pattern matching")
                            return valid_evaluations
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        st.warning(f"Pattern matching failed: {e}")
    
    # Method 3: Try to find and parse individual JSON objects
    try:
        import re
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
        entities = []
        for obj_str in json_objects:
            try:
                obj = json.loads(obj_str)
                if is_valid_evaluation_object(obj):
                    entities.append(obj)
            except:
                continue
        
        if entities:
            st.info(f"Recovered {len(entities)} evaluations from individual JSON objects")
            return entities
                
    except Exception:
        pass
    
    # Final fallback: Log error and return empty
    st.error(f"❌ Failed to parse evaluation response for batch {batch_idx if batch_idx is not None else 'unknown'}")
    
    # Show debugging info
    with st.expander(f"🔍 Debug Response Content (Batch {batch_idx})", expanded=False):
        st.text("Full raw response:")
        st.code(response_text, language="text")
        st.text(f"Response length: {len(response_text)} characters")
        st.text(f"First 200 chars: {repr(response_text[:200])}")
    
    return []


def validate_evaluation_structure(evaluations: list) -> list:
    """
    Validate and clean evaluation results structure.
    """
    valid_evaluations = []
    required_fields = ["entity_index", "current_text", "current_label", "is_correct", "recommendation"]
    
    for i, eval_result in enumerate(evaluations):
        if not isinstance(eval_result, dict):
            st.warning(f"Evaluation {i} is not a dictionary: {type(eval_result)}")
            continue
            
        # Check required fields
        missing_fields = [field for field in required_fields if field not in eval_result]
        if missing_fields:
            st.warning(f"Evaluation {i} missing fields: {missing_fields}")
            
            # Try to fill missing fields with defaults
            if 'entity_index' not in eval_result:
                eval_result['entity_index'] = i
            if 'current_text' not in eval_result:
                eval_result['current_text'] = 'Unknown'
            if 'current_label' not in eval_result:
                eval_result['current_label'] = 'Unknown'
            if 'is_correct' not in eval_result:
                eval_result['is_correct'] = False
            if 'recommendation' not in eval_result:
                eval_result['recommendation'] = 'manual_review'
        
        # Ensure proper data types
        try:
            eval_result['entity_index'] = int(eval_result.get('entity_index', i))
            eval_result['is_correct'] = bool(eval_result.get('is_correct', False))
        except (ValueError, TypeError):
            st.warning(f"Data type issues in evaluation {i}, using defaults")
            eval_result['entity_index'] = i
            eval_result['is_correct'] = False
        
        valid_evaluations.append(eval_result)
    
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
    for eval_idx in selected_indices:
        if eval_idx < len(evaluations):
            evaluation = evaluations[eval_idx]
            entity_idx = evaluation.get('entity_index')
           
            if entity_idx is None or entity_idx >= len(updated_entities):
                changes_made.append(f"Warning: Invalid entity index {entity_idx}")
                continue
               
            recommendation = evaluation.get('recommendation', '')
            current_text = updated_entities[entity_idx].get('text', 'Unknown')
           
            if recommendation == 'change_label' and evaluation.get('suggested_label'):
                # Change the label
                old_label = updated_entities[entity_idx].get('label', 'Unknown')
                new_label = evaluation['suggested_label']
                updated_entities[entity_idx]['label'] = new_label
                changes_made.append(f"Changed '{current_text}' from '{old_label}' to '{new_label}'")
               
            elif recommendation == 'delete':
                # Mark for deletion (will delete later)
                entities_to_delete.append(entity_idx)
                changes_made.append(f"Marked '{current_text}' for deletion")
   
    # Delete entities (in reverse order to maintain indices)
    if entities_to_delete:
        for entity_idx in sorted(set(entities_to_delete), reverse=True):
            if entity_idx < len(updated_entities):
                deleted_text = updated_entities[entity_idx].get('text', 'Unknown')
                deleted_label = updated_entities[entity_idx].get('label', 'Unknown')
                updated_entities.pop(entity_idx)
                changes_made.append(f"Deleted entity: '{deleted_text}' (label: '{deleted_label}')")
   
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
    
    try:
        # Method 1: Try direct JSON parsing first
        entities = json.loads(response_text)
        if isinstance(entities, list):
            # Validate entity structure
            valid_entities = []
            for ent in entities:
                if isinstance(ent, dict) and all(key in ent for key in ["start_char", "end_char", "text", "label"]):
                    valid_entities.append(ent)
                else:
                    st.warning(f"Invalid entity structure: {ent}")
            return valid_entities
        else:
            st.warning(f"Response is not a list: {type(entities)}")
            return []
            
    except json.JSONDecodeError:
        # Method 2: Try to extract JSON array from text
        try:
            first_bracket = response_text.find('[')
            last_bracket = response_text.rfind(']')
            
            if first_bracket == -1 or last_bracket == -1 or first_bracket >= last_bracket:
                raise ValueError("No valid JSON array found")
                
            json_str = response_text[first_bracket:last_bracket+1]
            entities = json.loads(json_str)
            
            # Validate entity keys
            valid_entities = []
            for ent in entities:
                if isinstance(ent, dict) and all(key in ent for key in ["start_char", "end_char", "text", "label"]):
                    valid_entities.append(ent)
                else:
                    st.warning(f"Invalid entity structure: {ent}")
            
            if len(valid_entities) != len(entities):
                st.warning(f"Some entities were invalid and filtered out")
            
            return valid_entities
            
        except (json.JSONDecodeError, ValueError) as e:
            # Method 3: Try to find and parse multiple JSON objects
            try:
                # Look for individual JSON objects
                import re
                json_objects = re.findall(r'\{[^{}]*\}', response_text)
                entities = []
                for obj_str in json_objects:
                    try:
                        obj = json.loads(obj_str)
                        if all(key in obj for key in ["start_char", "end_char", "text", "label"]):
                            entities.append(obj)
                    except:
                        continue
                
                if entities:
                    st.info(f"Recovered {len(entities)} entities from malformed response")
                    return entities
                    
            except Exception:
                pass
            
            # Final fallback: Log error and return empty
            st.error(f"Failed to parse LLM output JSON for chunk {chunk_index if chunk_index else 'unknown'}: {e}")
            st.error(f"Raw response preview: {response_text[:200]}...")
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
    
    for i, chunk in enumerate(chunks):
        with progress_container:
            # Clear previous progress display
            progress_container.empty()
            
            # Show current progress
            display_chunk_progress(i + 1, len(chunks), chunk, start_time)
            
            # Process the chunk
            with st.spinner(f"🤖 Calling {st.session_state.model_provider} API..."):
                prompt = build_annotation_prompt(tag_df, chunk)
                response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                entities = parse_llm_response(response)
                entities = aggregate_entities(entities, char_pos)
                all_entities.extend(entities)
                
                # Show chunk results
                st.success(f"✅ Chunk {i+1} completed! Found {len(entities)} entities.")
                
            char_pos += len(chunk) + 1  # +1 for newline or split char
    
    # Final summary
    total_time = time.time() - start_time
    st.balloons()
    st.success(f"🎉 All chunks processed in {total_time:.1f} seconds!")
    
    return all_entities
