# app.py

import streamlit as st
import pandas as pd
import io
import json
import streamlit as st
import pandas as pd
from prompts import build_annotation_prompt
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

def parse_llm_response(response_text: str):
    """
    Parse the JSON returned by LLM.
    Returns list of entities or empty list on error.
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
                
        return entities
    except Exception as e:
        st.error(f"Failed to parse LLM output JSON: {e}")
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

    # Create a progress bar and status container
    progress_bar = st.progress(0)
    status_container = st.container()
    
    with status_container:
        st.info(f"📄 Text split into {len(chunks)} chunks for processing...")
        
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
        
        prompt = build_annotation_prompt(tag_df, chunk)
        response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        entities = parse_llm_response(response)
        entities = aggregate_entities(entities, char_pos)
        all_entities.extend(entities)
        char_pos += len(chunk) + 1  # +1 for newline or split char

    # Final completion message
    progress_bar.progress(1.0)
    with status_container:
        st.success(f"✅ Processing complete! Analyzed {len(chunks)} chunks and found {len(all_entities)} entities.")

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
        confidence = ent.get("confidence", 0.5)
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
                # Include confidence in the tooltip
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

    # Append any remaining text after all entities
    highlighted.append(html.escape(text[last_pos:]))

    return ''.join(highlighted)


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
            )
            
            # Store results in session state
            st.session_state.annotated_entities = entities
            st.session_state.annotation_complete = True
            
            st.success(f"Annotation finished! Found {len(entities)} entities.")

        except Exception as e:
            st.error(f"Annotation failed: {e}")

# === Visual Highlight ===
st.subheader("🔍 Annotated Text Preview")

if 'annotated_entities' in st.session_state and st.session_state.annotated_entities:
    highlighted_html = highlight_text_with_entities(
    st.session_state.text_data,
    st.session_state.annotated_entities,
    st.session_state.label_colors
)

    styled_html = f"""
    <div style="font-family: Arial; font-size: 16px; line-height: 1.7;">
        {highlighted_html}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)
    

# Display and edit results (outside of button click)
import json
import pandas as pd
import streamlit as st

# Display and edit results (outside of button click)
if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    st.header("📝 Edit Annotations")
    
    # Convert to DataFrame for editing
    df_entities = pd.DataFrame(st.session_state.annotated_entities)

    # Add zero-based index as a column named "ID"
    df_entities.insert(0, "ID", range(len(df_entities)))
    
    # Ensure confidence column exists and is properly formatted
    if "confidence" not in df_entities.columns:
        df_entities["confidence"] = 0.5
    
    # Reorder columns to put confidence after label
    if "confidence" in df_entities.columns:
        cols = df_entities.columns.tolist()
        cols.remove("confidence")
        label_idx = cols.index("label") if "label" in cols else len(cols) - 1
        cols.insert(label_idx + 1, "confidence")
        df_entities = df_entities[cols]

    # Use st.column_config to customize columns
    column_config = {
        "ID": st.column_config.NumberColumn("ID", disabled=True),
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
    
    # Show editable table with row numbers, add unique key for Streamlit widget
    edited_df = st.data_editor(
        df_entities,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
        key="annotation_data_editor"
    )
    
    # Update session state with edited data (remove "ID" column before saving)
    updated_entities = edited_df.drop(columns=["ID"]).to_dict(orient="records")
    
    # Ensure confidence values are properly bounded
    for entity in updated_entities:
        if "confidence" in entity:
            try:
                entity["confidence"] = max(0.0, min(1.0, float(entity["confidence"])))
            except (ValueError, TypeError):
                entity["confidence"] = 0.5
    
    st.session_state.annotated_entities = updated_entities
    
    # Display summary statistics
    if updated_entities:
        confidences = [ent.get("confidence", 0.5) for ent in updated_entities]
        avg_confidence = sum(confidences) / len(confidences)
        st.info(f"📊 Summary: {len(updated_entities)} entities | Average confidence: {avg_confidence:.2f}")
    
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
    }
    json_str = json.dumps(output_json, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📥 Download Annotations as JSON", 
        data=json_str, 
        file_name="annotations.json", 
        mime="application/json",
        key="download_json_btn"
    )