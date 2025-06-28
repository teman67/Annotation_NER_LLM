import streamlit as st
import pandas as pd
import io
import json
import streamlit as st
import pandas as pd
from prompts_flat import build_annotation_prompt
from helper import (
    chunk_text,     # Function to chunk text into smaller pieces
    aggregate_entities,  # Function to aggregate entities with offsets
    display_chunk_progress,  # Function to display progress of chunk processing 
    display_processing_summary,  # Function to show processing summary
    generate_label_colors,  # Function to generate colors for labels
    get_token_recommendations,  # Function to get token recommendations based on chunk size
    validate_annotations_streamlit,  # Function to validate annotations
    fix_annotation_positions_streamlit,  # Function to fix annotation positions
    display_annotated_entities,  # Function to display annotated entities in a visually appealing way
    run_annotation_pipeline,  # Function to run the annotation pipeline
    clear_all_previous_data,  # Function to clear all previous data
    evaluate_annotations_with_llm,  # Function to evaluate annotations with LLM
    apply_evaluation_recommendations, # Function to apply evaluation recommendations
    run_annotation_pipeline,  # Function to run the annotation pipeline
)
from llm_clients import LLMClient
import html
import time
import streamlit.components.v1 as components
import colorsys
import hashlib

# ----- Page Setup -----
st.set_page_config(page_title="LLM-based Scientific Text Annotator", layout="wide")

# ----- Title and Description -----
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
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'evaluation_summary' not in st.session_state:
    st.session_state.evaluation_summary = {}

st.sidebar.header("🔐 API Configuration")

api_key = st.sidebar.text_input("Paste your API key", type="password")
model_provider = st.sidebar.selectbox("Choose LLM provider", ["OpenAI", "Claude"])

st.session_state.api_key = api_key
st.session_state.model_provider = model_provider

if model_provider == "OpenAI":
    model = st.sidebar.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"])
else:
    model = st.sidebar.selectbox("Claude model", ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Processing Parameters")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, step=0.05, 
                                help="Lower = more consistent, Higher = more creative")

chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 4000, 1000, step=100,
                              help="Size of text chunks to process separately")


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
            # Clear ALL previous data when starting new annotation
            clear_all_previous_data()  # This function we defined in step 3
            
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

            # DEBUG: Add comprehensive debugging (keep your existing debug code)
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

            # Show problematic entities (keep your existing debug code)
            problematic_entities = [e for e in st.session_state.annotated_entities 
                                if not all(key in e for key in ['start_char', 'end_char', 'text', 'label'])]

            if problematic_entities:
                with st.expander("⚠️ Problematic Entities (missing required fields)", expanded=True):
                    st.json(problematic_entities[:5])  # Show first 5

            # Check for entities with invalid positions (keep your existing debug code)
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

            # Show entity distribution by label (keep your existing debug code)
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

display_annotated_entities()
    
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

# === LLM Evaluation Section ===
if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    st.markdown("---")
    st.subheader("🤖 LLM Evaluation & Suggestions")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.write("Use LLM to evaluate whether annotated entities match their tag definitions and get suggestions for improvements.")
    
    with col2:
        if st.button("🤖 Evaluate Annotations", key="evaluate_annotations_btn"):
            if not st.session_state.api_key:
                st.error("❌ API key missing for evaluation")
            elif not st.session_state.annotated_entities:
                st.error("❌ No annotations to evaluate")
            elif st.session_state.tag_df is None:
                st.error("❌ Tag definitions missing")
            else:
                with st.spinner("🤖 LLM is evaluating your annotations..."):
                    try:
                        # Create LLM client
                        client = LLMClient(
                            api_key=st.session_state.api_key,
                            provider=st.session_state.model_provider,
                            model=model,
                        )
                        
                        # Run evaluation
                        evaluations = evaluate_annotations_with_llm(
                            st.session_state.annotated_entities,
                            st.session_state.tag_df,
                            client,
                            temperature=0.1,  # Low temperature for consistent evaluation
                            max_tokens=2000
                        )
                        
                        # Store results in session state
                        st.session_state.evaluation_results = evaluations
                        st.session_state.evaluation_complete = True
                        
                        # Calculate summary statistics
                        total_entities = len(st.session_state.annotated_entities)
                        correct_count = sum(1 for eval_result in evaluations if eval_result.get('is_correct', False))
                        change_recommendations = sum(1 for eval_result in evaluations if eval_result.get('recommendation') == 'change_label')
                        delete_recommendations = sum(1 for eval_result in evaluations if eval_result.get('recommendation') == 'delete')
                        
                        st.session_state.evaluation_summary = {
                            'total_entities': total_entities,
                            'evaluated_entities': len(evaluations),
                            'correct_count': correct_count,
                            'change_recommendations': change_recommendations,
                            'delete_recommendations': delete_recommendations
                        }
                        
                        st.success(f"✅ Evaluation completed! Analyzed {len(evaluations)} entities.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {e}")

# Display evaluation results if they exist (outside the button click)
if st.session_state.get('evaluation_complete') and st.session_state.get('evaluation_results'):
    
    # Display evaluation summary
    st.markdown("### 📊 Evaluation Summary")
    
    summary = st.session_state.evaluation_summary
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Total Entities", len(st.session_state.annotated_entities))  # Use current count
    with col_b:
        # Recalculate accuracy based on current entities and evaluation results
        current_correct = 0
        valid_evaluations = 0
        for eval_result in st.session_state.evaluation_results:
            entity_idx = eval_result.get('entity_index', -1)
            if 0 <= entity_idx < len(st.session_state.annotated_entities):  # Only count if entity still exists
                valid_evaluations += 1
                if eval_result.get('is_correct', False):
                    current_correct += 1
        
        accuracy = current_correct / valid_evaluations * 100 if valid_evaluations > 0 else 0
        st.metric("Correct", current_correct, delta=f"{accuracy:.1f}%")
    with col_c:
        # Count remaining change recommendations
        remaining_changes = sum(1 for eval_result in st.session_state.evaluation_results 
                               if eval_result.get('recommendation') == 'change_label' and 
                               0 <= eval_result.get('entity_index', -1) < len(st.session_state.annotated_entities))
        st.metric("Remaining Changes", remaining_changes)
    with col_d:
        # Count remaining delete recommendations
        remaining_deletes = sum(1 for eval_result in st.session_state.evaluation_results 
                               if eval_result.get('recommendation') == 'delete' and 
                               0 <= eval_result.get('entity_index', -1) < len(st.session_state.annotated_entities))
        st.metric("Remaining Deletions", remaining_deletes)
    
    # Display evaluation results table - FIXED VERSION
    st.markdown("### 📋 Evaluation Results & Recommendations")
    
    # Convert evaluation results to DataFrame for display - FIXED TO INCLUDE ALL ENTITIES
    eval_df_display = []
    
    # Create a mapping of all entities with their evaluation results
    entity_evaluation_map = {}
    for eval_result in st.session_state.evaluation_results:
        entity_idx = eval_result.get('entity_index', -1)
        if 0 <= entity_idx < len(st.session_state.annotated_entities):
            entity_evaluation_map[entity_idx] = eval_result
    
    # Process ALL entities in the current annotated_entities list
    for current_idx, entity in enumerate(st.session_state.annotated_entities):
        current_text = entity.get('text', '')
        current_label = entity.get('label', '')
        
        # Check if we have evaluation results for this entity
        if current_idx in entity_evaluation_map:
            eval_result = entity_evaluation_map[current_idx]
            
            # Check if this recommendation was already applied
            recommendation = eval_result.get('recommendation', '')
            is_applied = False
            
            if recommendation == 'change_label':
                suggested_label = eval_result.get('suggested_label', '')
                # If current label matches suggested label, recommendation was applied
                is_applied = (current_label == suggested_label)
            elif recommendation == 'delete':
                # If we're here, entity wasn't deleted, so not applied
                is_applied = False
            
            # Determine correctness - if recommendation was applied and it was change_label, now it's correct
            is_correct = eval_result.get('is_correct', False)
            if is_applied and recommendation == 'change_label':
                is_correct = True
            
            status = ''
            if is_applied:
                status = '✅ Applied'
            elif not is_correct:
                status = '❌ Needs Action'
            else:
                status = '✅ Correct'
            
            eval_df_display.append({
                'ID': current_idx,
                'Text': current_text,
                'Current Label': current_label,
                'Status': status,
                # 'Confidence': f"{eval_result.get('confidence', 0)*100:.0f}%" if eval_result.get('confidence') else 'N/A',
                'Recommendation': recommendation if not is_applied else 'Applied ✅',
                'Suggested Label': eval_result.get('suggested_label', '') or 'N/A',
                'Reasoning': eval_result.get('reasoning', '')[:300] + '...' if len(eval_result.get('reasoning', '')) > 300 else eval_result.get('reasoning', '')
            })
        else:
            # Entity has no evaluation result - this might happen if evaluation was incomplete
            eval_df_display.append({
                'ID': current_idx,
                'Text': current_text,
                'Current Label': current_label,
                'Status': '⚠️ Not Evaluated',
                # 'Confidence': 'N/A',
                'Recommendation': 'N/A',
                'Suggested Label': 'N/A',
                'Reasoning': 'No evaluation data available'
            })
    
    if eval_df_display:
        eval_display_df = pd.DataFrame(eval_df_display)
        
        # Show evaluation table
        st.dataframe(eval_display_df, use_container_width=True, height=400)
        
        # Show debug information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write(f"**Total entities in annotated_entities:** {len(st.session_state.annotated_entities)}")
            st.write(f"**Total evaluation results:** {len(st.session_state.evaluation_results)}")
            st.write(f"**Entities displayed in table:** {len(eval_df_display)}")
            
            # Show entity indices in evaluation results
            eval_indices = [eval_result.get('entity_index', -1) for eval_result in st.session_state.evaluation_results]
            st.write(f"**Entity indices in evaluation results:** {sorted(eval_indices)}")
            
            # Show which entities have no evaluation
            evaluated_indices = set(eval_indices)
            all_indices = set(range(len(st.session_state.annotated_entities)))
            missing_indices = all_indices - evaluated_indices
            if missing_indices:
                st.write(f"**Entities missing evaluation:** {sorted(missing_indices)}")
            else:
                st.write("**All entities have evaluation results**")
        
        # Filter for actionable recommendations (NOT YET APPLIED) - FIXED
        actionable_evals = []
        for eval_result in st.session_state.evaluation_results:
            entity_idx = eval_result.get('entity_index', -1)
            
            # Skip if entity index is invalid
            if not (0 <= entity_idx < len(st.session_state.annotated_entities)):
                continue
                
            recommendation = eval_result.get('recommendation', '')
            
            if recommendation == 'delete':
                # Delete recommendations are always actionable if entity exists
                actionable_evals.append(eval_result)
            elif recommendation == 'change_label':
                # Change recommendations are actionable if not already applied
                current_entity = st.session_state.annotated_entities[entity_idx]
                current_label = current_entity.get('label', '')
                suggested_label = eval_result.get('suggested_label', '')
                
                # Only actionable if current label != suggested label
                if current_label != suggested_label:
                    actionable_evals.append(eval_result)
        
        if actionable_evals:
            st.markdown("### 🔧 Apply Recommendations")
            
            # Create selection options for REMAINING recommendations only
            selection_options = []
            
            for eval_result in actionable_evals:
                entity_idx = eval_result.get('entity_index', -1)
                # Find the evaluation index for this eval_result
                eval_idx = next((i for i, er in enumerate(st.session_state.evaluation_results) if er == eval_result), -1)
                
                if 0 <= entity_idx < len(st.session_state.annotated_entities):
                    current_entity = st.session_state.annotated_entities[entity_idx]
                    current_text = current_entity.get('text', eval_result.get('current_text', ''))
                    
                    if eval_result.get('recommendation') == 'delete':
                        action = "DELETE" 
                    else:
                        action = f"CHANGE to '{eval_result.get('suggested_label')}'"
                    
                    option_text = f"[Entity {entity_idx}] '{current_text}' → {action}"
                    selection_options.append((eval_idx, option_text))
            
            # Multiselect for recommendations to apply
            selected_recommendations = st.multiselect(
                "Select recommendations to apply:",
                options=[idx for idx, _ in selection_options],
                format_func=lambda x: next(text for idx, text in selection_options if idx == x),
                key="selected_eval_recommendations"
            )
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("✅ Apply Selected", disabled=not selected_recommendations, key="apply_recommendations_btn"):
                    if selected_recommendations:
                        try:
                            # Apply recommendations (including automatic deletions)
                            updated_entities, changes_made = apply_evaluation_recommendations(
                                st.session_state.annotated_entities,
                                st.session_state.evaluation_results,
                                selected_recommendations
                            )
                            
                            # Update session state
                            st.session_state.annotated_entities = updated_entities
                            
                            # Update editable dataframe if it exists
                            if 'editable_entities_df' in st.session_state:
                                try:
                                    df_updated = pd.DataFrame(updated_entities)
                                    if not df_updated.empty:
                                        df_updated.insert(0, "ID", range(len(df_updated)))
                                        st.session_state.editable_entities_df = df_updated
                                    else:
                                        st.session_state.editable_entities_df = pd.DataFrame()
                                except Exception as df_error:
                                    st.warning(f"Could not update editable dataframe: {df_error}")
                            
                            # Update evaluation results to reflect the changes
                            # Remove evaluation results for deleted entities and update indices
                            remaining_evaluation_results = []
                            entity_index_mapping = {}  # old_index -> new_index
                            
                            # Create mapping for entities that weren't deleted
                            new_idx = 0
                            for old_idx in range(len(st.session_state.annotated_entities) + len([e for e in st.session_state.evaluation_results if e.get('recommendation') == 'delete' and e.get('entity_index', -1) in [st.session_state.evaluation_results[i].get('entity_index') for i in selected_recommendations]])):
                                # Check if this entity was deleted
                                was_deleted = any(
                                    st.session_state.evaluation_results[sel_idx].get('entity_index') == old_idx and 
                                    st.session_state.evaluation_results[sel_idx].get('recommendation') == 'delete'
                                    for sel_idx in selected_recommendations
                                )
                                
                                if not was_deleted:
                                    entity_index_mapping[old_idx] = new_idx
                                    new_idx += 1
                            
                            # Update evaluation results with new indices
                            for eval_result in st.session_state.evaluation_results:
                                old_entity_idx = eval_result.get('entity_index', -1)
                                if old_entity_idx in entity_index_mapping:
                                    eval_result['entity_index'] = entity_index_mapping[old_entity_idx]
                                    remaining_evaluation_results.append(eval_result)
                                # If not in mapping, entity was deleted, so don't include this evaluation result
                            
                            st.session_state.evaluation_results = remaining_evaluation_results
                            
                            # Show success message with changes
                            st.success(f"✅ Applied {len(selected_recommendations)} recommendations!")
                            
                            if changes_made:
                                with st.expander("📝 Changes Made", expanded=True):
                                    for change in changes_made:
                                        st.write(f"• {change}")
                            
                            # Clear the selection to avoid re-applying
                            if 'selected_eval_recommendations' in st.session_state:
                                del st.session_state['selected_eval_recommendations']
                            
                            # Rerun to refresh the display
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Failed to apply recommendations: {e}")
                            import traceback
                            st.error(traceback.format_exc())  # For debugging
            
            with col2:
                if selected_recommendations:
                    st.info(f"💡 {len(selected_recommendations)} recommendation(s) selected for application.")
                else:
                    st.info("💡 Select recommendations above to apply them.")
        
        else:
            st.success("🎉 All recommendations have been applied or no actionable recommendations remain!")
    
    else:
        st.info("ℹ️ No evaluation results to display (all entities may have been processed).")

# Option to clear evaluation results
if st.button("🧹 Clear Evaluation Results", key="clear_eval_results_btn"):
    st.session_state.evaluation_results = []
    st.session_state.evaluation_complete = False
    st.session_state.evaluation_summary = {}
    if 'selected_eval_recommendations' in st.session_state:
        del st.session_state['selected_eval_recommendations']
    st.rerun()


if st.session_state.get("annotation_complete") and st.session_state.get("annotated_entities"):
    st.markdown("---")
    st.header("💾 Export Results")
    
    # Build comprehensive output JSON
    output_json = {
        "text": st.session_state.get("text_data", ""),
        "entities": st.session_state.annotated_entities,
        "metadata": {
            "total_entities": len(st.session_state.annotated_entities),
            "annotation_timestamp": pd.Timestamp.now().isoformat(),
            "model_provider": st.session_state.get("model_provider", ""),
            "model": model if 'model' in locals() else "",
            "processing_parameters": {
                "temperature": temperature if 'temperature' in locals() else 0.1,
                "chunk_size": chunk_size if 'chunk_size' in locals() else 1000,
                "max_tokens": max_tokens if 'max_tokens' in locals() else 1000
            }
        }
    }
    
    # Add evaluation data if available
    if st.session_state.get('evaluation_complete') and st.session_state.get('evaluation_results'):
        output_json["evaluation"] = {
            "evaluation_results": st.session_state.evaluation_results,
            "evaluation_summary": st.session_state.evaluation_summary,
            "evaluation_timestamp": pd.Timestamp.now().isoformat()
        }
        # st.info("📊 Export includes LLM evaluation results and recommendations.")
    
    # Add validation data if available
    if st.session_state.get('validation_results'):
        output_json["validation"] = {
            "validation_results": st.session_state.validation_results,
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        st.info("✅ Export includes validation results.")
    
    # Add fix data if available
    if st.session_state.get('fix_results'):
        output_json["position_fixes"] = {
            "fix_results": st.session_state.fix_results,
            "fix_timestamp": pd.Timestamp.now().isoformat()
        }
        # st.info("🔧 Export includes position fix results.")
    
    json_str = json.dumps(output_json, indent=2, ensure_ascii=False)
       
    # # Download button
    # st.download_button(
    #     "📥 Download Complete Annotations with Metadata", 
    #     data=json_str, 
    #     file_name=f"annotations_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json", 
    #     mime="application/json",
    #     key="download_complete_json_btn"
    # )
    
    # Optional: Also provide basic annotations-only export
    basic_json = {
        "text": st.session_state.get("text_data", ""),
        "entities": st.session_state.annotated_entities,
    }
    basic_json_str = json.dumps(basic_json, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📥 Download Annotated Text", 
        data=basic_json_str, 
        file_name=f"annotations_basic_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json", 
        mime="application/json",
        key="download_basic_json_btn"
    )
    
    st.markdown("---")

    # Optional clear all button
    if st.button("🧹 Clear All Annotations"):
        st.session_state.annotated_entities = []
        st.session_state.editable_entities_df = pd.DataFrame()
        st.session_state.annotation_complete = False
        st.rerun()