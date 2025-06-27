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