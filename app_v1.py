# app.py

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

    styled_html = f"""
    <style>
        .annotation-container {{
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.7;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        
        .annotation-container span[data-tooltip] {{
            position: relative;
            cursor: help;
        }}
        
        .annotation-container span[data-tooltip]:hover::after {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: normal;
            white-space: nowrap;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .annotation-container span[data-tooltip]:hover::before {{
            content: '';
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(100%);
            border: 6px solid transparent;
            border-top-color: #333;
            z-index: 1000;
        }}
    </style>
    <div class="annotation-container">
        {highlighted_html}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)
    

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

    # Optional clear all button
    if st.button("🧹 Clear All Annotations"):
        st.session_state.annotated_entities = []
        st.session_state.editable_entities_df = pd.DataFrame()
        st.session_state.annotation_complete = False
        st.rerun()


# Download annotated JSON (outside of button click)
if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    st.header("💾 Export Results")
    
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