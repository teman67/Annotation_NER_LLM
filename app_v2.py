# app.py

import streamlit as st
import pandas as pd
import io
import json
import streamlit as st
import pandas as pd
from prompts import build_annotation_prompt, build_nested_annotation_prompt
from llm_clients import LLMClient
import html

# ----- Page Setup -----
st.set_page_config(page_title="LLM-based Scientific Text Annotator", layout="wide")

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


st.title("🔬 Scientific Text Annotator with LLMs")
st.markdown("Use OpenAI or Claude models to annotate scientific text with custom tag definitions.")

# ----- Session State Setup -----
if 'text_data' not in st.session_state:
    st.session_state.text_data = ""
if 'tag_df' not in st.session_state:
    st.session_state.tag_df = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'model_provider' not in st.session_state:
    st.session_state.model_provider = "OpenAI"
if 'annotated_entities' not in st.session_state:
    st.session_state.annotated_entities = []
if 'annotation_complete' not in st.session_state:
    st.session_state.annotation_complete = False
if 'nested_annotation_mode' not in st.session_state:
    st.session_state.nested_annotation_mode = False

# ----- Sidebar -----
st.sidebar.header("🔐 API Configuration")

api_key = st.sidebar.text_input("Paste your API key", type="password")
model_provider = st.sidebar.selectbox("Choose LLM provider", ["OpenAI", "Claude"])

st.session_state.api_key = api_key
st.session_state.model_provider = model_provider

if model_provider == "OpenAI":
    model = st.sidebar.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"])
else:
    model = st.sidebar.selectbox("Claude model", ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"])

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)
chunk_size = st.sidebar.slider("Chunk size (in characters)", 500, 4000, 800, step=100)
max_tokens = st.sidebar.slider("Max tokens per response", 200, 6000, 1000, step=100)

st.sidebar.markdown("---")
clean_text = st.sidebar.checkbox("Clean text input (remove weird characters)", value=True)

# NEW: Nested annotation mode toggle
st.sidebar.markdown("---")
st.sidebar.header("🎯 Annotation Mode")
nested_mode = st.sidebar.checkbox("Enable Nested Annotations", value=False, 
                                  help="Allow entities to contain other entities within them")
st.session_state.nested_annotation_mode = nested_mode

if nested_mode:
    st.sidebar.info("📝 Nested mode will identify hierarchical relationships between entities")

# ----- File Upload -----
st.header("📄 Upload Scientific Text")
uploaded_text = st.file_uploader("Upload a `.txt` file or paste below", type=["txt"])

if uploaded_text:
    text = uploaded_text.read().decode("utf-8", errors="ignore")
    if clean_text:
        text = ''.join(c for c in text if c.isprintable())
    st.session_state.text_data = text

text_area_input = st.text_area("Or paste text here:", st.session_state.text_data, height=200)

if text_area_input:
    st.session_state.text_data = text_area_input

# ----- CSV Tag Upload -----
st.header("🏷️ Upload Tag Set CSV")
uploaded_csv = st.file_uploader("Upload a `.csv` file with `tag_name`, `definition`, and `examples` columns", type=["csv"])

if uploaded_csv:
    try:
        tag_df = pd.read_csv(uploaded_csv)
        required_cols = {"tag_name", "definition", "examples"}
        if not required_cols.issubset(tag_df.columns):
            st.error("CSV file must include columns: tag_name, definition, examples.")
        else:
            st.session_state.tag_df = tag_df
            st.success("✅ Tag file loaded successfully!")
            st.dataframe(tag_df)
            st.session_state.label_colors = generate_label_colors(tag_df['tag_name'].unique())
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")

# ----- Input Validation -----
st.header("🧠 Ready to Annotate?")

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

def parse_llm_response(response_text: str, nested_mode: bool = False):
    """
    Parse the JSON returned by LLM.
    Returns list of entities or empty list on error.
    For nested mode, handles hierarchical structure.
    """
    try:
        # Some LLMs may wrap JSON inside extra text, try to extract JSON array by first [ and last ]
        first_bracket = response_text.find('[')
        last_bracket = response_text.rfind(']')
        json_str = response_text[first_bracket:last_bracket+1]
        entities = json.loads(json_str)
        
        # Validate entity keys and add default confidence if missing
        required_keys = {"start_char", "end_char", "text", "label"}
        for ent in entities:
            if not set(ent.keys()) >= required_keys:
                st.warning("Warning: Some entities missing required keys")
            # Add default confidence score if not present
            if "confidence" not in ent:
                ent["confidence"] = 0.5  # Default confidence
            # Ensure confidence is a float between 0 and 1
            try:
                ent["confidence"] = max(0.0, min(1.0, float(ent["confidence"])))
            except (ValueError, TypeError):
                ent["confidence"] = 0.5  # Default if invalid
            
            # Handle nested entities
            if nested_mode and "nested_entities" in ent:
                # Recursively process nested entities
                nested_entities = ent["nested_entities"]
                if isinstance(nested_entities, list):
                    for nested_ent in nested_entities:
                        if "confidence" not in nested_ent:
                            nested_ent["confidence"] = 0.5
                        try:
                            nested_ent["confidence"] = max(0.0, min(1.0, float(nested_ent["confidence"])))
                        except (ValueError, TypeError):
                            nested_ent["confidence"] = 0.5
                            
        return entities
    except Exception as e:
        # st.error(f"Failed to parse LLM output JSON: {e}")
        return []

def aggregate_entities(all_entities, offset):
    """
    Adjust entity character positions by offset (chunk start position in full text).
    Also adjusts nested entities if present.
    """
    for ent in all_entities:
        ent['start_char'] += offset
        ent['end_char'] += offset
        
        # Adjust nested entities if present
        if 'nested_entities' in ent and isinstance(ent['nested_entities'], list):
            for nested_ent in ent['nested_entities']:
                nested_ent['start_char'] += offset
                nested_ent['end_char'] += offset
                
    return all_entities

def run_annotation_pipeline(text, tag_df, client, temperature, max_tokens, chunk_size, nested_mode=False):
    """
    1. Chunk the text
    2. For each chunk, generate prompt and call LLM
    3. Parse and adjust entities with offset
    4. Aggregate and return full list of entities
    """
    chunks = chunk_text(text, chunk_size)
    all_entities = []
    char_pos = 0

    # Create a progress bar and status container
    progress_bar = st.progress(0)
    status_container = st.container()
    
    with status_container:
        mode_text = "nested" if nested_mode else "flat"
        st.info(f"📄 Text split into {len(chunks)} chunks for {mode_text} annotation processing...")
        
        # Show chunk overview
        chunk_info = []
        for i, chunk in enumerate(chunks):
            chunk_info.append({
                "Chunk": i + 1,
                "Size (chars)": len(chunk),
                "Preview": chunk[:50] + "..." if len(chunk) > 50 else chunk
            })
        
        with st.expander("📋 View Chunk Details", expanded=False):
            st.dataframe(pd.DataFrame(chunk_info), use_container_width=True)

    for i, chunk in enumerate(chunks):
        # Update progress
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        
        # Create a nice status message with emoji and metrics
        with status_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Chunk", f"{i+1}/{len(chunks)}")
            with col2:
                st.metric("Chunk Size", f"{len(chunk)} chars")
            with col3:
                st.metric("Progress", f"{progress:.0%}")
        
        # Show current chunk preview in an expandable section
        with status_container:
            with st.expander(f"🔍 Preview Chunk {i+1}", expanded=False):
                st.text_area(
                    f"Chunk {i+1} Content:",
                    chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    height=150,
                    disabled=True
                )
        
        # Choose appropriate prompt based on mode
        if nested_mode:
            prompt = build_nested_annotation_prompt(tag_df, chunk)
        else:
            prompt = build_annotation_prompt(tag_df, chunk)
            
        response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        entities = parse_llm_response(response, nested_mode)
        entities = aggregate_entities(entities, char_pos)
        all_entities.extend(entities)
        char_pos += len(chunk) + 1  # +1 for newline or split char

    # Final completion message
    progress_bar.progress(1.0)
    with status_container:
        # Count total entities including nested ones
        total_entities = len(all_entities)
        nested_count = sum(len(ent.get('nested_entities', [])) for ent in all_entities)
        if nested_count > 0:
            st.success(f"✅ Processing complete! Analyzed {len(chunks)} chunks and found {total_entities} entities with {nested_count} nested entities.")
        else:
            st.success(f"✅ Processing complete! Analyzed {len(chunks)} chunks and found {total_entities} entities.")

    return all_entities

def highlight_text_with_entities(text: str, entities: list, label_colors: dict, nested_mode: bool = False) -> str:
    """
    Enhanced highlighting function that handles nested entities with visual hierarchy.
    """
    import html
    
    if not nested_mode:
        # Use original highlighting for flat mode
        return highlight_text_flat(text, entities, label_colors)
    
    # For nested mode, we need to handle hierarchical highlighting
    used_positions = set()
    highlighted = []
    last_pos = 0

    # Sort entities by start position, then by length (longer first for proper nesting)
    sorted_entities = sorted(entities, key=lambda x: (x.get("start_char", 0), -(x.get("end_char", 0) - x.get("start_char", 0))))

    for ent in sorted_entities:
        start = ent.get("start_char", 0)
        end = ent.get("end_char", 0)
        span = ent["text"]
        label = ent["label"]
        confidence = ent.get("confidence", 0.5)
        color = label_colors.get(label, "#e0e0e0")
        
        # Check if this entity overlaps with already used positions
        if any(i in used_positions for i in range(start, end)):
            continue
            
        # Add text before this entity
        if start > last_pos:
            highlighted.append(html.escape(text[last_pos:start]))
        
        # Create the highlighted span with nested entities
        nested_entities = ent.get("nested_entities", [])
        if nested_entities:
            # Handle nested highlighting within this entity
            entity_text = text[start:end]
            nested_html = highlight_nested_entities(entity_text, nested_entities, label_colors, start)
            tooltip_text = f"{html.escape(label)} (confidence: {confidence:.2f}, {len(nested_entities)} nested)"
            highlighted.append(
                f'<span style="background-color: {color}; border: 2px solid {color}; padding: 2px; margin: 1px; border-radius: 3px; font-weight: bold; cursor: help;" title="{tooltip_text}">'
                f'{nested_html}</span>'
            )
        else:
            tooltip_text = f"{html.escape(label)} (confidence: {confidence:.2f})"
            highlighted.append(
                f'<mark style="background-color: {color}; font-weight: bold; cursor: help;" title="{tooltip_text}">'
                f'{html.escape(span)}</mark>'
            )
        
        # Mark positions as used
        used_positions.update(range(start, end))
        last_pos = end

    # Append any remaining text
    if last_pos < len(text):
        highlighted.append(html.escape(text[last_pos:]))

    return ''.join(highlighted)

def highlight_nested_entities(entity_text: str, nested_entities: list, label_colors: dict, parent_start: int) -> str:
    """
    Highlight nested entities within a parent entity.
    """
    import html
    
    if not nested_entities:
        return html.escape(entity_text)
    
    highlighted = []
    last_pos = 0
    
    # Sort nested entities by their relative position within the parent
    sorted_nested = sorted(nested_entities, key=lambda x: x.get("start_char", 0) - parent_start)
    
    for nested_ent in sorted_nested:
        # Calculate relative position within parent entity
        relative_start = nested_ent.get("start_char", 0) - parent_start
        relative_end = nested_ent.get("end_char", 0) - parent_start
        
        # Ensure positions are within bounds
        if relative_start < 0 or relative_end > len(entity_text):
            continue
            
        # Add text before this nested entity
        if relative_start > last_pos:
            highlighted.append(html.escape(entity_text[last_pos:relative_start]))
        
        # Add the nested entity with different styling
        nested_span = entity_text[relative_start:relative_end]
        nested_label = nested_ent["label"]
        nested_confidence = nested_ent.get("confidence", 0.5)
        nested_color = label_colors.get(nested_label, "#e0e0e0")
        
        tooltip_text = f"{html.escape(nested_label)} (confidence: {nested_confidence:.2f}, nested)"
        highlighted.append(
            f'<span style="background-color: {nested_color}; border: 1px dashed {nested_color}; padding: 1px; margin: 0px; border-radius: 2px; font-style: italic; cursor: help;" title="{tooltip_text}">'
            f'{html.escape(nested_span)}</span>'
        )
        
        last_pos = relative_end
    
    # Add remaining text
    if last_pos < len(entity_text):
        highlighted.append(html.escape(entity_text[last_pos:]))
    
    return ''.join(highlighted)

def highlight_text_flat(text: str, entities: list, label_colors: dict) -> str:
    """
    Original flat highlighting function.
    """
    import html
    used_positions = set()
    highlighted = []
    last_pos = 0

    sorted_entities = sorted(entities, key=lambda x: x.get("start_char", 0))

    for ent in sorted_entities:
        span = ent["text"]
        label = ent["label"]
        confidence = ent.get("confidence", 0.5)
        color = label_colors.get(label, "#e0e0e0")

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
                tooltip_text = f"{html.escape(label)} (confidence: {confidence:.2f})"
                highlighted.append(
                    f'<mark style="background-color: {color}; font-weight: bold; cursor: help;" title="{tooltip_text}">'
                    f'{html.escape(span)}</mark>'
                )
                used_positions.update(range(idx, idx + len(span)))
                last_pos = idx + len(span)
                found = True
                break

        if not found:
            continue

    highlighted.append(html.escape(text[last_pos:]))
    return ''.join(highlighted)

def flatten_entities_for_display(entities: list) -> list:
    """
    Flatten nested entities into a single list for display in the data editor.
    Adds a 'parent_id' field to nested entities.
    """
    flattened = []
    
    for i, entity in enumerate(entities):
        # Add the main entity
        flat_entity = entity.copy()
        flat_entity['entity_id'] = i
        flat_entity['parent_id'] = None
        flat_entity['is_nested'] = False
        
        # Remove nested_entities for display (we'll show them separately)
        if 'nested_entities' in flat_entity:
            nested_count = len(flat_entity['nested_entities'])
            flat_entity['nested_count'] = nested_count
            del flat_entity['nested_entities']
        else:
            flat_entity['nested_count'] = 0
            
        flattened.append(flat_entity)
        
        # Add nested entities if they exist
        if 'nested_entities' in entity:
            for j, nested_entity in enumerate(entity['nested_entities']):
                nested_flat = nested_entity.copy()
                nested_flat['entity_id'] = f"{i}.{j}"
                nested_flat['parent_id'] = i
                nested_flat['is_nested'] = True
                nested_flat['nested_count'] = 0
                flattened.append(nested_flat)
    
    return flattened

def reconstruct_nested_entities(flattened_df: pd.DataFrame) -> list:
    """
    Reconstruct nested entity structure from flattened DataFrame.
    """
    entities = []
    nested_entities_map = {}
    
    # Group nested entities by parent_id
    for _, row in flattened_df.iterrows():
        row_dict = row.to_dict()
        
        if row_dict.get('is_nested', False) and row_dict.get('parent_id') is not None:
            parent_id = row_dict['parent_id']
            if parent_id not in nested_entities_map:
                nested_entities_map[parent_id] = []
            
            # Clean up the nested entity dict
            nested_entity = {k: v for k, v in row_dict.items() 
                           if k not in ['entity_id', 'parent_id', 'is_nested', 'nested_count']}
            nested_entities_map[parent_id].append(nested_entity)
    
    # Reconstruct main entities
    entity_id = 0
    for _, row in flattened_df.iterrows():
        row_dict = row.to_dict()
        
        if not row_dict.get('is_nested', False):
            # Clean up the main entity dict
            entity = {k: v for k, v in row_dict.items() 
                     if k not in ['entity_id', 'parent_id', 'is_nested', 'nested_count']}
            
            # Add nested entities if they exist
            if entity_id in nested_entities_map:
                entity['nested_entities'] = nested_entities_map[entity_id]
            
            entities.append(entity)
            entity_id += 1
    
    return entities

# === Streamlit UI ===

if st.button("🔍 Run Annotation", key="run_annotation_btn"):
    if not st.session_state.api_key:
        st.error("❌ API key missing")
    elif not st.session_state.text_data:
        st.error("❌ Text missing")
    elif st.session_state.tag_df is None:
        st.error("❌ Tag CSV missing")
    else:
        try:
            client = LLMClient(
                api_key=st.session_state.api_key,
                provider=st.session_state.model_provider,
                model=model,
            )
            entities = run_annotation_pipeline(
                text=st.session_state.text_data,
                tag_df=st.session_state.tag_df,
                client=client,
                temperature=temperature,
                max_tokens=max_tokens,
                chunk_size=chunk_size,
                nested_mode=st.session_state.nested_annotation_mode
            )
            
            # Store results in session state
            st.session_state.annotated_entities = entities
            st.session_state.annotation_complete = True
            
            # Count nested entities for display
            nested_count = sum(len(ent.get('nested_entities', [])) for ent in entities)
            if nested_count > 0:
                st.success(f"Annotation finished! Found {len(entities)} entities with {nested_count} nested entities.")
            else:
                st.success(f"Annotation finished! Found {len(entities)} entities.")

        except Exception as e:
            st.error(f"Annotation failed: {e}")

# === Visual Highlight ===
st.subheader("🔍 Annotated Text Preview")

if 'annotated_entities' in st.session_state and st.session_state.annotated_entities:
    highlighted_html = highlight_text_with_entities(
        st.session_state.text_data,
        st.session_state.annotated_entities,
        st.session_state.label_colors,
        nested_mode=st.session_state.nested_annotation_mode
    )

    # Add legend for nested annotations
    if st.session_state.nested_annotation_mode:
        st.markdown("""
        **Legend:** 
        - 🔲 **Solid border**: Main entities
        - 📦 **Dashed border**: Nested entities (italic text)
        - Hover over highlights to see confidence scores
        """)

    styled_html = f"""
    <div style="font-family: Arial; font-size: 16px; line-height: 1.7;">
        {highlighted_html}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)

# Display and edit results (outside of button click)
if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    st.header("📝 Edit Annotations")
    
    # Handle nested vs flat display
    if st.session_state.nested_annotation_mode:
        # Flatten entities for display
        flattened_entities = flatten_entities_for_display(st.session_state.annotated_entities)
        df_entities = pd.DataFrame(flattened_entities)
        
        # Add styling column for better visualization
        df_entities['display_style'] = df_entities.apply(
            lambda row: "📦 Nested" if row.get('is_nested', False) else "🔲 Main", axis=1
        )
    else:
        # Convert to DataFrame for editing (flat mode)
        df_entities = pd.DataFrame(st.session_state.annotated_entities)
        df_entities['display_style'] = "🔲 Main"

    # Add zero-based index as a column named "ID"
    df_entities.insert(0, "ID", range(len(df_entities)))
    
    # Ensure confidence column exists and is properly formatted
    if "confidence" not in df_entities.columns:
        df_entities["confidence"] = 0.5
    
    # Reorder columns for better display
    column_order = ["ID", "display_style", "text", "label", "confidence", "start_char", "end_char"]
    if st.session_state.nested_annotation_mode:
        column_order.extend(["parent_id", "nested_count"])
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df_entities.columns]
    other_columns = [col for col in df_entities.columns if col not in existing_columns]
    df_entities = df_entities[existing_columns + other_columns]

    # Use st.column_config to customize columns
    column_config = {
        "ID": st.column_config.NumberColumn("ID", disabled=True),
        "display_style": st.column_config.TextColumn("Type", disabled=True, width="small"),
        "confidence": st.column_config.NumberColumn(
            "Confidence Score",
            help="Confidence score between 0.0 and 1.0",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f"
        ),
        "start_char": st.column_config.NumberColumn("Start Char"),
        "end_char": st.column_config.NumberColumn("End Char"),
        "text": st.column_config.TextColumn("Text", width="medium"),
        "label": st.column_config.TextColumn("Label", width="medium"),
    }
    
    if st.session_state.nested_annotation_mode:
        column_config.update({
            "parent_id": st.column_config.NumberColumn("Parent ID", disabled=True),
            "nested_count": st.column_config.NumberColumn("Nested Count", disabled=True),
        })
    
    # Show editable table
    edited_df = st.data_editor(
        df_entities,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
        key="annotation_data_editor_nested" if st.session_state.nested_annotation_mode else "annotation_data_editor_flat"
    )
    
    # Update session state with edited data
    if st.session_state.nested_annotation_mode:
        # Reconstruct nested structure
        # Remove display columns before reconstruction
        clean_df = edited_df.drop(columns=["ID", "display_style"], errors="ignore")
        updated_entities = reconstruct_nested_entities(clean_df)
    else:
        # Simple flat update
        updated_entities = edited_df.drop(columns=["ID", "display_style"], errors="ignore").to_dict(orient="records")
    
    # Ensure confidence values are properly bounded
    def fix_confidence(entities_list):
        for entity in entities_list:
            if "confidence" in entity:
                try:
                    entity["confidence"] = max(0.0, min(1.0, float(entity["confidence"])))
                except (ValueError, TypeError):
                    entity["confidence"] = 0.5
            
            # Fix nested entities confidence too
            if "nested_entities" in entity:
                fix_confidence(entity["nested_entities"])
    
    fix_confidence(updated_entities)
    st.session_state.annotated_entities = updated_entities
    
    # Display summary statistics
    if updated_entities:
        main_count = len(updated_entities)
        nested_count = sum(len(ent.get("nested_entities", [])) for ent in updated_entities)
        
        # Calculate average confidence
        all_confidences = []
        for entity in updated_entities:
            all_confidences.append(entity.get("confidence", 0.5))
            for nested_ent in entity.get("nested_entities", []):
                all_confidences.append(nested_ent.get("confidence", 0.5))
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        if nested_count > 0:
            st.info(f"📊 Summary: {main_count} main entities, {nested_count} nested entities | Average confidence: {avg_confidence:.2f}")
        else:
            st.info(f"📊 Summary: {main_count} entities | Average confidence: {avg_confidence:.2f}")
    
    # Clear annotations button
    if st.button("🗑️ Clear Annotations", key="clear_annotations_btn"):
        st.session_state.annotated_entities = []
        st.session_state.annotation_complete = False
        st.rerun()

# Download annotated JSON (outside of button click)
if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    st.header("💾 Export Results")
    
    output_json = {
        "text": st.session_state.get("text_data", ""),
        "entities": st.session_state.annotated_entities,
        "annotation_mode": "nested" if st.session_state.nested_annotation_mode else "flat",
        "metadata": {
            "total_entities": len(st.session_state.annotated_entities),
            "nested_entities": sum(len(ent.get('nested_entities', [])) for ent in st.session_state.annotated_entities),
            "model_used": model,
            "provider": st.session_state.model_provider,
            "temperature": temperature,
            "chunk_size": chunk_size
        }
    }
    json_str = json.dumps(output_json, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📥 Download Annotations as JSON", 
        data=json_str, 
        file_name="annotations.json", 
        mime="application/json",
        key="download_json_btn"
    )
    
    # Show JSON preview
    with st.expander("🔍 Preview JSON Structure", expanded=False):
        st.json(output_json)