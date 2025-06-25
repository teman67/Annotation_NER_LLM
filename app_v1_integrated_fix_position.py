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

st.sidebar.header("🔐 API Configuration")

api_key = st.sidebar.text_input("Paste your API key", type="password")
model_provider = st.sidebar.selectbox("Choose LLM provider", ["OpenAI", "Claude"])

st.session_state.api_key = api_key
st.session_state.model_provider = model_provider

if model_provider == "OpenAI":
    model = st.sidebar.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"])
else:
    model = st.sidebar.selectbox("Claude model", ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Processing Parameters")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, step=0.05, 
                                help="Lower = more consistent, Higher = more creative")

chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 4000, 1000, step=100,
                              help="Size of text chunks to process separately")

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

min_tokens, max_tokens_limit, default_tokens = get_token_recommendations(chunk_size)

max_tokens = st.sidebar.slider(
    "Max tokens per response", 
    min_tokens, 
    max_tokens_limit, 
    default_tokens, 
    step=50,
    help=f"Recommended: {default_tokens} tokens for {chunk_size} character chunks"
)

# Show the relationship
st.sidebar.info(f"""
**Current Settings:**
- Chunk: {chunk_size:,} chars (~{chunk_size//4:,} tokens input)
- Response: {max_tokens:,} tokens max output
- Ratio: {max_tokens/(chunk_size//4):.1f}x output/input
""")

# Warning if settings seem problematic
if max_tokens > chunk_size // 2:
    st.sidebar.warning("⚠️ Max tokens seems very high for this chunk size")
elif max_tokens < chunk_size // 20:
    st.sidebar.warning("⚠️ Max tokens might be too low - responses may get cut off")

st.sidebar.markdown("---")
clean_text = st.sidebar.checkbox("Clean text input (remove weird characters)", value=True)

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

def aggregate_entities(all_entities, offset):
    """
    Adjust entity character positions by offset (chunk start position in full text).
    """
    for ent in all_entities:
        ent['start_char'] += offset
        ent['end_char'] += offset
    return all_entities

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


# === Show Processing Summary ===
if st.session_state.text_data and st.session_state.tag_df is not None:
    display_processing_summary(
        st.session_state.text_data, 
        st.session_state.tag_df, 
        chunk_size, 
        temperature, 
        max_tokens, 
        model_provider, 
        model
    )

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
            # Clear previous annotation results when starting new annotation
            st.session_state.annotated_entities = []
            st.session_state.annotation_complete = False
            if 'editable_entities_df' in st.session_state:
                del st.session_state.editable_entities_df

            # Clear validation and fix results
            if 'validation_results' in st.session_state:
                del st.session_state.validation_results
            if 'fix_results' in st.session_state:
                del st.session_state.fix_results
            
            st.markdown("### 🚀 Starting Annotation Process")
            
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
            )
            
            # Store results in session state
            st.session_state.annotated_entities = entities
            st.session_state.annotation_complete = True

            # DEBUG: Add comprehensive debugging
            st.markdown("### 🔍 Annotation Information")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw Entities from LLM", len(st.session_state.annotated_entities))
            with col2:
                # Check for duplicates
                entity_texts = [e.get('text', '') for e in st.session_state.annotated_entities]
                unique_texts = len(set(entity_texts))
                st.metric("Unique Entity Texts", unique_texts)
            with col3:
                # Check for invalid entities
                valid_entities = [e for e in st.session_state.annotated_entities 
                                if all(key in e for key in ['start_char', 'end_char', 'text', 'label'])]
                st.metric("Valid Entities", len(valid_entities))

            # Show problematic entities
            problematic_entities = [e for e in st.session_state.annotated_entities 
                                if not all(key in e for key in ['start_char', 'end_char', 'text', 'label'])]

            if problematic_entities:
                with st.expander("⚠️ Problematic Entities (missing required fields)", expanded=True):
                    st.json(problematic_entities[:5])  # Show first 5

            # Check for entities with invalid positions
            invalid_pos_entities = []
            text_length = len(st.session_state.text_data)
            for e in st.session_state.annotated_entities:
                start = e.get('start_char', 0)
                end = e.get('end_char', 0)
                if start < 0 or end > text_length or start >= end:
                    invalid_pos_entities.append(e)

            if invalid_pos_entities:
                with st.expander("⚠️ Entities with Invalid Positions", expanded=True):
                    st.json(invalid_pos_entities[:5])

            # Show entity distribution by label
            if st.session_state.annotated_entities:
                entity_df_debug = pd.DataFrame(st.session_state.annotated_entities)
                label_counts = entity_df_debug['label'].value_counts()
                
                with st.expander("📊 Entity Distribution by Label", expanded=False):
                    st.bar_chart(label_counts)
            
            st.success(f"🎯 Annotation completed! Found {len(entities)} entities total.")

        except Exception as e:
            st.error(f"❌ Annotation failed: {e}")

# === Visual Highlight ===
st.subheader("🔍 Annotated Text Preview")

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
    

# Display and edit results (outside of button click)
import json
import pandas as pd
import streamlit as st

if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    st.header("📝 Edit Annotations")

    # Initialize or reload dataframe from session state, including ID column
    if "editable_entities_df" not in st.session_state:
        # FIXED: Filter out invalid entities before creating DataFrame
        valid_entities = []
        for e in st.session_state.annotated_entities:
            # Check if entity has all required fields
            required_fields = ['start_char', 'end_char', 'text', 'label']
            if all(field in e and e[field] is not None for field in required_fields):
                # Additional validation
                if (isinstance(e['start_char'], (int, float)) and 
                    isinstance(e['end_char'], (int, float)) and
                    e['start_char'] >= 0 and 
                    e['end_char'] > e['start_char'] and
                    isinstance(e['text'], str) and 
                    len(e['text'].strip()) > 0):
                    valid_entities.append(e)
                else:
                    st.warning(f"Filtered out invalid entity: {e}")
            else:
                st.warning(f"Filtered out entity missing required fields: {e}")
        
        if len(valid_entities) != len(st.session_state.annotated_entities):
            st.warning(f"⚠️ Filtered out {len(st.session_state.annotated_entities) - len(valid_entities)} invalid entities")
            st.session_state.annotated_entities = valid_entities
        
        try:
            df_entities = pd.DataFrame(valid_entities)
            if not df_entities.empty:
                df_entities.insert(0, "ID", range(len(df_entities)))
                st.session_state.editable_entities_df = df_entities
                st.success(f"✅ Created DataFrame with {len(df_entities)} valid entities")
            else:
                st.error("❌ No valid entities to display")
                st.session_state.editable_entities_df = pd.DataFrame()
        except Exception as e:
            st.error(f"Error creating DataFrame: {e}")
            st.session_state.editable_entities_df = pd.DataFrame()
    else:
        df_entities = st.session_state.editable_entities_df

    # Show editable table, disabled ID column
    edited_df = st.data_editor(
        df_entities,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
        },
        key="annotation_data_editor",
        disabled=["ID"],
        hide_index=True,
    )

    # Save edits back to session state (except ID column)
    st.session_state.editable_entities_df = edited_df

    # Multiselect options from current df
    to_delete_ids = st.multiselect(
        "Select annotation ID(s) to remove:",
        options=edited_df["ID"].tolist() if not edited_df.empty else [],
        key="delete_selected_ids"
    )

    if st.button("🗑 Remove Selected Annotations"):
        if to_delete_ids:
            # Filter out rows to delete
            filtered_df = edited_df[~edited_df["ID"].isin(to_delete_ids)].reset_index(drop=True)
            # Re-assign ID sequentially
            filtered_df["ID"] = range(len(filtered_df))

            # Update session state dataframe
            st.session_state.editable_entities_df = filtered_df

            # Also update annotated_entities (without ID)
            st.session_state.annotated_entities = filtered_df.drop(columns=["ID"]).to_dict(orient="records")

            st.success(f"Removed {len(to_delete_ids)} annotation(s).")
            st.rerun()
        else:
            st.warning("Please select annotation ID(s) to remove.")

    # FIXED: Only update annotated_entities when user actually made changes
    # Check if the edited_df is different from what we started with
    if not edited_df.equals(df_entities):
        st.session_state.annotated_entities = edited_df.drop(columns=["ID"]).to_dict(orient="records")

    
# Download annotated JSON (outside of button click)
# Add these functions to your Streamlit app (after the existing functions)

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

# Replace the existing "💾 Export Results" section with this enhanced version:

# Replace the validation button section with this code:
st.markdown("---")
# Download annotated JSON (outside of button click)
if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    
    
    # Add validation and fixing section
    st.subheader("🔍 Validate & Fix Annotations Position")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Validate Annotations", key="validate_btn"):
            with st.spinner("Validating annotations..."):
                validation_results = validate_annotations_streamlit(
                    st.session_state.text_data, 
                    st.session_state.annotated_entities
                )
                
                # Store validation results in session state
                st.session_state.validation_results = validation_results
    
    # Display validation results if they exist (outside the button click)
    if st.session_state.get('validation_results'):
        validation_results = st.session_state.validation_results
        
        # Display validation summary
        st.markdown("### 📊 Validation Results")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Entities", validation_results['total_entities'])
        with col_b:
            st.metric("Correct", validation_results['correct_entities'], 
                     delta=f"{validation_results['correct_entities']/validation_results['total_entities']*100:.1f}%")
        with col_c:
            st.metric("Errors", len(validation_results['errors']))
        with col_d:
            st.metric("Warnings", len(validation_results['warnings']))
        
        # Show errors if any
        if validation_results['errors']:
            st.error(f"❌ Found {len(validation_results['errors'])} annotation errors!")
            
            with st.expander("📋 View Error Details", expanded=False):
                error_data = []
                for error in validation_results['errors'][:100]:  # Show first 100 errors
                    error_data.append({
                        "Index": error['entity_index'],
                        "Expected Text": error['expected_text'],
                        "Actual Text": error.get('actual_text', 'N/A'),
                        "Position": f"[{error['start_char']}:{error['end_char']}]",
                        "Label": error['label'],
                        "Error": error.get('error', 'Text mismatch')
                    })
                
                if error_data:
                    st.dataframe(pd.DataFrame(error_data), use_container_width=True)
                
                if len(validation_results['errors']) > 100:
                    st.info(f"Showing first 100 of {len(validation_results['errors'])} errors.")
        
        # Show warnings if any
        if validation_results['warnings']:
            st.warning(f"⚠️ Found {len(validation_results['warnings'])} warnings!")
    
            with st.expander("⚠️ View Warning Details", expanded=False):
                for i, warning in enumerate(validation_results['warnings']):
                    if warning.get('type') == 'overlap':
                        st.write(f"**Overlap {i+1}:**")
                        st.write(f"- Entity 1: '{warning['entity1']['text']}' [{warning['entity1']['start_char']}:{warning['entity1']['end_char']}]")
                        st.write(f"- Entity 2: '{warning['entity2']['text']}' [{warning['entity2']['start_char']}:{warning['entity2']['end_char']}]")
                    else:
                        st.write(f"**Zero-length annotation {i+1}:** {warning}")
                
        if not validation_results['errors']:
            st.success("✅ All position of the annotations are valid!")
    
    with col2:
        # Only show fix button if validation has been run and there are errors
        if (st.session_state.get('validation_results') and 
            st.session_state.validation_results.get('errors')):
            
            fix_strategy = st.selectbox(
                "Fix Strategy", 
                ["first" , "closest"],
                help="closest: Choose position closest to original | first: Use first occurrence found"
            )
            
            if st.button("🔧 Fix Annotations", key="fix_btn"):
                with st.spinner("Fixing annotations..."):
                    fixed_entities, fix_stats = fix_annotation_positions_streamlit(
                        st.session_state.text_data,
                        st.session_state.annotated_entities,
                        strategy=fix_strategy
                    )
                    
                    # Update session state with fixed entities
                    st.session_state.annotated_entities = fixed_entities
                    
                    # Update the editable dataframe if it exists
                    if 'editable_entities_df' in st.session_state:
                        try:
                            df_fixed = pd.DataFrame(fixed_entities)
                            if not df_fixed.empty:
                                df_fixed.insert(0, "ID", range(len(df_fixed)))
                                st.session_state.editable_entities_df = df_fixed
                        except:
                            pass  # If DataFrame creation fails, just skip
                    
                    # Store fix results in session state
                    st.session_state.fix_results = fix_stats
                    
                    # Clear validation results to allow re-validation
                    if 'validation_results' in st.session_state:
                        del st.session_state.validation_results
                    
                    st.rerun()
    
    # Display fix results if they exist (outside the button click)
    if st.session_state.get('fix_results'):
        fix_stats = st.session_state.fix_results
        
        # Display fix results
        st.markdown("### 🔧 Fix Results")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total", fix_stats['total'])
        with col_b:
            st.metric("Already Correct", fix_stats['already_correct'])
        with col_c:
            st.metric("Fixed", fix_stats['fixed'], 
                     delta=f"{fix_stats['fixed']/fix_stats['total']*100:.1f}%")
        with col_d:
            st.metric("Unfixable", fix_stats['unfixable'])
        
        success_rate = (fix_stats['already_correct'] + fix_stats['fixed']) / fix_stats['total'] * 100
        
        if fix_stats['fixed'] > 0:
            st.success(f"🎉 Successfully fixed {fix_stats['fixed']} annotations! Overall success rate: {success_rate:.1f}%")
        
        if fix_stats['unfixable'] > 0:
            st.warning(f"⚠️ Could not fix {fix_stats['unfixable']} annotations. Manual review may be needed.")
        
        if fix_stats['multiple_matches'] > 0:
            st.info(f"ℹ️ {fix_stats['multiple_matches']} annotations had multiple possible positions. Used fix strategy.")
        
        st.info("💡 You can now re-validate to check if all issues were resolved!")
        
        # Clear fix results after displaying
        if st.button("Clear Fix Results", key="clear_fix_results"):
            del st.session_state.fix_results
            st.rerun()
    
    st.markdown("---")
    st.header("💾 Export Results")
    # Original download functionality
    output_json = {
        "text": st.session_state.get("text_data", ""),
        "entities": st.session_state.annotated_entities,
    }
    json_str = json.dumps(output_json, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📥 Download Annotations as JSON", 
        data=json_str, 
        file_name="annotations.json", 
        mime="application/json",
        key="download_json_btn"
    )
    st.markdown("---")
    

    # Optional clear all button
    if st.button("🧹 Clear All Annotations"):
        st.session_state.annotated_entities = []
        st.session_state.editable_entities_df = pd.DataFrame()
        st.session_state.annotation_complete = False
        st.rerun()