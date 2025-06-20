import streamlit as st
import pandas as pd
import json
import openai
import anthropic
from typing import List, Dict, Any
import re
import time
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Scientific Text Annotator",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'tag_definitions' not in st.session_state:
    st.session_state.tag_definitions = None
if 'current_annotation_preview' not in st.session_state:
    st.session_state.current_annotation_preview = None
if 'raw_response' not in st.session_state:
    st.session_state.raw_response = None

class ScientificTextAnnotator:
    def __init__(self, openai_api_key: str = None, claude_api_key: str = None):
        self.openai_client = None
        self.claude_client = None
        
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        if claude_api_key:
            self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
    
    def create_annotation_prompt(self, text: str, tag_definitions: pd.DataFrame) -> str:
        """Create a comprehensive prompt for NER annotation"""
        
        # Build tag definitions section
        tag_definitions_str = ""
        for _, row in tag_definitions.iterrows():
            tag_definitions_str += f"""
**{row['tag_name']}**: {row['definition']}
Examples: {row['examples']}
"""
        
        prompt = f"""You are an expert scientific text annotator. Your task is to perform Named Entity Recognition (NER) on scientific texts using the provided tag set.

## TAG DEFINITIONS:
{tag_definitions_str}

## ANNOTATION INSTRUCTIONS:
1. Identify ALL entities in the text that match the defined categories
2. For each entity, determine the EXACT character positions (start and end indices)
3. Be precise with entity boundaries - include articles, prepositions only if they're part of the entity
4. Handle nested entities by choosing the most specific applicable tag
5. If an entity could belong to multiple categories, choose the most specific one
6. Only annotate entities that clearly fit the definitions provided

## CRITICAL: OUTPUT FORMAT
You MUST return ONLY a valid JSON object with this exact structure. Do not include any other text before or after the JSON:

{{
  "text": "original text here",
  "entities": [
    {{
      "text": "entity text",
      "label": "TAG_NAME",
      "start": start_position,
      "end": end_position,
      "confidence": confidence_score_0_to_1
    }}
  ],
  "metadata": {{
    "total_entities": number_of_entities,
    "annotation_timestamp": "current_timestamp"
  }}
}}

## TEXT TO ANNOTATE:
{text}

JSON Response:"""
        
        return prompt
    
    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract and parse JSON from API response with multiple strategies"""
        
        # Strategy 1: Try to parse the entire response as JSON
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for JSON object using regex (most common case)
        json_patterns = [
            r'\{.*\}',  # Basic JSON object
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    # If the pattern has a capture group, use it; otherwise use the full match
                    json_text = match if isinstance(match, str) else match
                    return json.loads(json_text.strip())
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Try to find JSON by looking for opening and closing braces
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                try:
                    json_text = response_text[start_idx:end_idx]
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 4: Try to create a minimal valid response from any detected entities
        # This is a fallback if we can detect some structure but can't parse full JSON
        try:
            # Look for entity-like patterns in the text
            entity_patterns = re.findall(r'"text":\s*"([^"]+)".*?"label":\s*"([^"]+)"', response_text, re.DOTALL)
            if entity_patterns:
                entities = []
                for i, (entity_text, label) in enumerate(entity_patterns):
                    entities.append({
                        "text": entity_text,
                        "label": label,
                        "start": 0,  # Default values since we can't extract positions
                        "end": len(entity_text),
                        "confidence": 0.8
                    })
                
                return {
                    "text": "Error in position extraction - entities found but positions may be incorrect",
                    "entities": entities,
                    "metadata": {
                        "total_entities": len(entities),
                        "annotation_timestamp": datetime.now().isoformat(),
                        "parsing_method": "fallback_entity_extraction"
                    }
                }
        except Exception:
            pass
        
        # If all strategies fail, return None
        return None
    
    def count_tokens_estimate(self, text: str) -> int:
        """Rough estimate of token count (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def chunk_text(self, text: str, max_tokens: int = 3000) -> List[str]:
        """Split text into chunks that fit within token limits"""
        estimated_tokens = self.count_tokens_estimate(text)
        
        if estimated_tokens <= max_tokens:
            return [text]
        
        # Split by sentences to maintain context
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if self.count_tokens_estimate(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def annotate_with_openai(self, text: str, tag_definitions: pd.DataFrame, 
                           model: str = "gpt-4", temperature: float = 0.1) -> Dict[str, Any]:
        """Annotate text using OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI API key not provided")
        
        # Check token limits and chunk if necessary
        estimated_tokens = self.count_tokens_estimate(text)
        max_input_tokens = 50000 if 'gpt-4' in model else 10000  # Conservative limits
        
        if estimated_tokens > max_input_tokens:
            st.warning(f"Text is large ({estimated_tokens} tokens). Chunking into smaller parts...")
            chunks = self.chunk_text(text, max_input_tokens)
            all_entities = []
            offset = 0
            
            for i, chunk in enumerate(chunks):
                st.info(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_result = self._process_chunk_openai(chunk, tag_definitions, model, temperature)
                
                if chunk_result and 'entities' in chunk_result:
                    # Adjust entity positions for chunk offset
                    for entity in chunk_result['entities']:
                        entity['start'] += offset
                        entity['end'] += offset
                    all_entities.extend(chunk_result['entities'])
                
                offset += len(chunk) + 1  # +1 for space between chunks
                time.sleep(1)  # Rate limiting
            
            return {
                'text': text,
                'entities': all_entities,
                'metadata': {
                    'total_entities': len(all_entities),
                    'model': model,
                    'provider': 'openai',
                    'annotation_timestamp': datetime.now().isoformat(),
                    'chunked': True,
                    'total_chunks': len(chunks)
                }
            }
        
        return self._process_chunk_openai(text, tag_definitions, model, temperature)
    
    def _process_chunk_openai(self, text: str, tag_definitions: pd.DataFrame, 
                             model: str, temperature: float, preview_mode: bool = False) -> Dict[str, Any]:
        """Process a single chunk with OpenAI"""
        prompt = self.create_annotation_prompt(text, tag_definitions)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise scientific text annotator. You MUST return only valid JSON without any additional text or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=4000  # Reduced to stay within limits
            )
            
            result = response.choices[0].message.content
            
            # Store raw response for preview mode
            if preview_mode:
                st.session_state.raw_response = result
            
            # Use improved JSON extraction
            parsed_result = self.extract_json_from_response(result)
            
            if parsed_result:
                # Add metadata if not present
                if 'metadata' not in parsed_result:
                    parsed_result['metadata'] = {}
                
                parsed_result['metadata'].update({
                    'model': model,
                    'provider': 'openai',
                    'annotation_timestamp': datetime.now().isoformat()
                })
                return parsed_result
            else:
                # Log the problematic response for debugging
                st.error(f"Failed to parse JSON response. Raw response: {result[:500]}...")
                raise ValueError("Could not parse JSON from response")
                    
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return None
    
    def annotate_with_claude(self, text: str, tag_definitions: pd.DataFrame, 
                           model: str = "claude-sonnet-4-20250514", temperature: float = 0.1) -> Dict[str, Any]:
        """Annotate text using Claude API"""
        if not self.claude_client:
            raise ValueError("Claude API key not provided")
        
        # Claude has much larger context window, but still check for very large texts
        estimated_tokens = self.count_tokens_estimate(text)
        max_input_tokens = 150000  # Conservative limit for Claude
        
        if estimated_tokens > max_input_tokens:
            st.warning(f"Text is very large ({estimated_tokens} tokens). Chunking into smaller parts...")
            chunks = self.chunk_text(text, max_input_tokens)
            all_entities = []
            offset = 0
            
            for i, chunk in enumerate(chunks):
                st.info(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_result = self._process_chunk_claude(chunk, tag_definitions, model, temperature)
                
                if chunk_result and 'entities' in chunk_result:
                    # Adjust entity positions for chunk offset
                    for entity in chunk_result['entities']:
                        entity['start'] += offset
                        entity['end'] += offset
                    all_entities.extend(chunk_result['entities'])
                
                offset += len(chunk) + 1  # +1 for space between chunks
                time.sleep(1)  # Rate limiting
            
            return {
                'text': text,
                'entities': all_entities,
                'metadata': {
                    'total_entities': len(all_entities),
                    'model': model,
                    'provider': 'claude',
                    'annotation_timestamp': datetime.now().isoformat(),
                    'chunked': True,
                    'total_chunks': len(chunks)
                }
            }
        
        return self._process_chunk_claude(text, tag_definitions, model, temperature)
    
    def _process_chunk_claude(self, text: str, tag_definitions: pd.DataFrame, 
                             model: str, temperature: float, preview_mode: bool = False) -> Dict[str, Any]:
        """Process a single chunk with Claude"""
        prompt = self.create_annotation_prompt(text, tag_definitions)
        
        try:
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=8000,  # Increased for Claude's higher limits
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text
            
            # Store raw response for preview mode
            if preview_mode:
                st.session_state.raw_response = result
            
            # Use improved JSON extraction
            parsed_result = self.extract_json_from_response(result)
            
            if parsed_result:
                # Add metadata if not present
                if 'metadata' not in parsed_result:
                    parsed_result['metadata'] = {}
                
                parsed_result['metadata'].update({
                    'model': model,
                    'provider': 'claude',
                    'annotation_timestamp': datetime.now().isoformat()
                })
                return parsed_result
            else:
                # Log the problematic response for debugging
                st.error(f"Failed to parse JSON response. Raw response: {result[:500]}...")
                raise ValueError("Could not parse JSON from response")
                    
        except Exception as e:
            st.error(f"Claude API Error: {str(e)}")
            return None

def validate_csv_format(df: pd.DataFrame) -> bool:
    """Validate that the CSV has required columns"""
    required_columns = ['tag_name', 'definition', 'examples']
    return all(col in df.columns for col in required_columns)

def display_annotation_preview(annotation_result: Dict[str, Any]) -> bool:
    """Display annotation preview and return whether user wants to proceed"""
    if not annotation_result:
        return False
    
    st.subheader("🔍 Annotation Preview")
    st.info("Review the annotations below. You can approve them or make changes before saving.")
    
    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entities Found", annotation_result['metadata']['total_entities'])
    with col2:
        st.metric("Model Used", annotation_result['metadata']['model'])
    with col3:
        st.metric("Provider", annotation_result['metadata']['provider'])
    
    # Display text with highlighted entities
    st.subheader("📝 Annotated Text Preview")
    text = annotation_result['text']
    entities = sorted(annotation_result['entities'], key=lambda x: x['start'])
    
    # Create highlighted text with yellow background for all entities
    highlighted_text = ""
    last_end = 0
    
    for entity in entities:
        # Add text before entity
        highlighted_text += text[last_end:entity['start']]
        
        # Add highlighted entity with yellow background
        highlighted_text += f"""<span style='background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; margin: 1px; border: 1px solid #ddd;' title='Label: {entity["label"]}, Confidence: {entity.get("confidence", "N/A")}'>{entity['text']}</span>"""
        last_end = entity['end']
    
    # Add remaining text
    highlighted_text += text[last_end:]
    
    st.markdown(highlighted_text, unsafe_allow_html=True)
    
    # Show entity legend - removed color variations since all are now yellow
    with st.expander("🎨 Entity Type Legend"):
        legend_cols = st.columns(4)
        unique_labels = list(set([entity['label'] for entity in entities]))
        
        for i, label in enumerate(unique_labels):
            col = legend_cols[i % 4]
            col.markdown(f"""<span style='background-color: #ffeb3b; padding: 2px 8px; border-radius: 3px; margin: 2px; display: inline-block;'>{label}</span>""", unsafe_allow_html=True)
    
    # Display entities table
    st.subheader("📊 Detected Entities Details")
    if entities:
        entities_df = pd.DataFrame(entities)
        # Add row numbers for reference
        entities_df.index = entities_df.index + 1
        st.dataframe(entities_df, use_container_width=True)
        
        # Entity statistics
        label_counts = entities_df['label'].value_counts()
        st.subheader("📈 Entity Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(label_counts)
        
        with col2:
            for label, count in label_counts.items():
                percentage = (count / len(entities)) * 100
                st.metric(f"{label}", f"{count} ({percentage:.1f}%)")
    else:
        st.info("No entities detected in the text.")
    
    # Show raw response if available (for debugging)
    if hasattr(st.session_state, 'raw_response') and st.session_state.raw_response:
        with st.expander("🔧 Raw API Response (for debugging)"):
            st.code(st.session_state.raw_response, language='json')
    
    # Action buttons
    st.subheader("Next Steps")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        approve = st.button("✅ Approve & Save Annotations", type="primary", use_container_width=True)
    
    with col2:
        edit = st.button("✏️ Edit Annotations", use_container_width=True)
    
    with col3:
        discard = st.button("❌ Discard & Start Over", use_container_width=True)
    
    # Add this to session state initialization (add to the existing session state checks)
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False

    if approve:
        return True
    elif edit:
        st.session_state.edit_mode = True
        st.rerun()
    elif discard:
        st.session_state.current_annotation_preview = None
        st.session_state.raw_response = None
        st.session_state.edit_mode = False
        st.rerun()

    # Show edit interface when in edit mode
    if st.session_state.edit_mode:
        st.info("💡 **Edit Mode**: You can manually adjust the entities table below, then click 'Apply Changes'.")
        
        # Allow editing of entities
        st.subheader("✏️ Edit Annotations")
        edited_entities = st.data_editor(
            entities_df,
            num_rows="dynamic",
            use_container_width=True,
            key="entity_editor"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Apply Changes"):
                # Update the annotation result with edited entities
                annotation_result['entities'] = edited_entities.to_dict('records')
                annotation_result['metadata']['total_entities'] = len(edited_entities)
                annotation_result['metadata']['manually_edited'] = True
                st.session_state.current_annotation_preview = annotation_result
                st.session_state.edit_mode = False
                st.success("✅ Changes applied successfully!")
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel Edit"):
                st.session_state.edit_mode = False
                st.rerun()

    return False

def main():
    st.title("🔬 Scientific Text Annotator")
    st.markdown("Upload your tag definitions and annotate scientific texts using OpenAI or Claude models.")
    
    # Sidebar for API keys and settings
    with st.sidebar:
        st.header("Configuration")
        
        # API Keys
        openai_key = st.text_input("OpenAI API Key", type="password")
        claude_key = st.text_input("Claude API Key", type="password")
        
        # Model selection
        st.subheader("Model Settings")
        provider = st.selectbox("Choose Provider", ["OpenAI", "Claude"])
        
        if provider == "OpenAI":
            model = st.selectbox("OpenAI Model", ["gpt-4o-mini","gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
            st.info("Token limits: GPT-4 (8K), GPT-4-turbo/4o (128K total, 16K output)")
        else:
            model = st.selectbox("Claude Model", ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-opus-4-20250514"])
            st.info("Token limits: Claude (200K+ input, 8K output)")
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        # Batch processing settings
        st.subheader("Batch Processing")
        batch_delay = st.slider("Delay between requests (seconds)", 0.5, 5.0, 1.0, 0.5)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📋 Setup", "🔍 Single Text", "📚 Batch Processing"])
    
    with tab1:
        st.header("Setup Tag Definitions")
        
        # Upload CSV file
        uploaded_file = st.file_uploader(
            "Upload Tag Definitions CSV",
            type=['csv'],
            help="CSV should contain columns: tag_name, definition, examples"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if validate_csv_format(df):
                    st.session_state.tag_definitions = df
                    st.success("✅ Tag definitions loaded successfully!")
                    
                    # Display the loaded definitions
                    st.subheader("Loaded Tag Definitions")
                    st.dataframe(df, use_container_width=True)
                    
                    # Show example CSV format
                    with st.expander("View Expected CSV Format"):
                        example_data = {
                            'tag_name': ['PROTEIN', 'DISEASE', 'CHEMICAL', 'GENE'],
                            'definition': [
                                'Proteins, enzymes, and protein complexes',
                                'Medical conditions, diseases, and disorders',
                                'Chemical compounds and molecules',
                                'Genes and genetic elements'
                            ],
                            'examples': [
                                'insulin, hemoglobin, p53 protein, cytochrome c',
                                'diabetes, cancer, Alzheimer\'s disease, hypertension',
                                'glucose, ATP, dopamine, acetylcholine',
                                'BRCA1, TP53, APOE, insulin gene'
                            ]
                        }
                        example_df = pd.DataFrame(example_data)
                        st.dataframe(example_df)
                        
                        # Download example CSV
                        csv_buffer = io.StringIO()
                        example_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "Download Example CSV",
                            csv_buffer.getvalue(),
                            "example_tag_definitions.csv",
                            "text/csv"
                        )
                else:
                    st.error("❌ CSV must contain columns: tag_name, definition, examples")
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    with tab2:
        st.header("Annotate Single Text")
        
        if st.session_state.tag_definitions is None:
            st.warning("⚠️ Please upload tag definitions first in the Setup tab.")
            return
        
        # Check if we have a preview to show
        if st.session_state.current_annotation_preview:
            # Show preview and handle user decision
            approved = display_annotation_preview(st.session_state.current_annotation_preview)
            
            if approved:
                # User approved the annotations, save them
                st.session_state.annotations.append(st.session_state.current_annotation_preview)
                
                # Show success message and download option
                st.success("✅ Annotations saved successfully!")
                
                # Download JSON
                json_str = json.dumps(st.session_state.current_annotation_preview, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
                
                # Clear the preview
                # st.session_state.current_annotation_preview = None
                # st.session_state.raw_response = None
                
        else:
            # Show the annotation input interface
            st.markdown("### 📝 Enter Text to Annotate")
            
            # Text input
            text_input = st.text_area(
                "Enter scientific text to annotate:",
                height=200,
                placeholder="Paste your scientific text here..."
            )
            
            # Show text statistics
            if text_input:
                word_count = len(text_input.split())
                char_count = len(text_input)
                est_tokens = char_count // 4
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", word_count)
                with col2:
                    st.metric("Characters", char_count)
                with col3:
                    st.metric("Est. Tokens", est_tokens)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Generate Annotation Preview", type="primary"):
                    if not text_input.strip():
                        st.error("Please enter text to annotate.")
                    elif not (openai_key or claude_key):
                        st.error("Please provide at least one API key.")
                    else:
                        # Initialize annotator
                        annotator = ScientificTextAnnotator(openai_key, claude_key)
                        
                        with st.spinner("Generating annotation preview..."):
                            try:
                                if provider == "OpenAI":
                                    result = annotator._process_chunk_openai(
                                        text_input, st.session_state.tag_definitions, model, temperature, preview_mode=True
                                    )
                                else:
                                    result = annotator._process_chunk_claude(
                                        text_input, st.session_state.tag_definitions, model, temperature, preview_mode=True
                                    )
                                
                                if result:
                                    st.session_state.current_annotation_preview = result
                                    st.rerun()
                                
                            except Exception as e:
                                st.error(f"Annotation failed: {str(e)}")
            
            with col2:
                if st.button("🗑️ Clear All Results"):
                    st.session_state.annotations = []
                    st.session_state.current_annotation_preview = None
                    st.session_state.raw_response = None
                    st.rerun()
            
            # Show previous annotations if any
            if st.session_state.annotations:
                st.markdown("---")
                st.subheader("📚 Previous Annotations")
                
                for i, annotation in enumerate(reversed(st.session_state.annotations)):
                    with st.expander(f"Annotation {len(st.session_state.annotations) - i}: {annotation['metadata']['total_entities']} entities found"):
                        # Show preview of the text (first 200 characters)
                        preview_text = annotation['text'][:200] + "..." if len(annotation['text']) > 200 else annotation['text']
                        st.text(preview_text)
                        
                        # Show some stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Entities", annotation['metadata']['total_entities'])
                        with col2:
                            st.metric("Model", annotation['metadata']['model'])
                        with col3:
                            st.metric("Provider", annotation['metadata']['provider'])
                        
                        # Download button for this annotation
                        json_str = json.dumps(annotation, indent=2)
                        st.download_button(
                            "📥 Download",
                            json_str,
                            f"annotation_{i+1}.json",
                            "application/json",
                            key=f"download_{i}"
                        )
    
    with tab3:
        st.header("Batch Processing")
        
        if st.session_state.tag_definitions is None:
            st.warning("⚠️ Please upload tag definitions first in the Setup tab.")
            return
        
        # Upload multiple texts
        text_files = st.file_uploader(
            "Upload text files for batch processing",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload multiple .txt files to process in batch"
        )
        
        if text_files:
            st.info(f"Loaded {len(text_files)} files for processing")
            
            if st.button("🚀 Start Batch Processing", type="primary"):
                if not (openai_key or claude_key):
                    st.error("Please provide at least one API key.")
                else:
                    annotator = ScientificTextAnnotator(openai_key, claude_key)
                    batch_results = []
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file in enumerate(text_files):
                        try:
                            # Read file content
                            text_content = file.read().decode('utf-8')
                            
                            status_text.text(f"Processing {file.name}...")
                            
                            # Annotate
                            if provider == "OpenAI":
                                result = annotator.annotate_with_openai(
                                    text_content, st.session_state.tag_definitions, model, temperature
                                )
                            else:
                                result = annotator.annotate_with_claude(
                                    text_content, st.session_state.tag_definitions, model, temperature
                                )
                            
                            if result:
                                result['metadata']['source_file'] = file.name
                                batch_results.append(result)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(text_files))
                            
                            # Delay between requests
                            if i < len(text_files) - 1:
                                time.sleep(batch_delay)
                                
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                    
                    status_text.text("Batch processing completed!")
                    
                    if batch_results:
                        st.success(f"✅ Successfully processed {len(batch_results)} files")
                        
                        # Display summary
                        st.subheader("Batch Results Summary")
                        summary_data = []
                        for result in batch_results:
                            summary_data.append({
                                'File': result['metadata']['source_file'],
                                'Entities Found': result['metadata']['total_entities'],
                                'Model': result['metadata']['model'],
                                'Provider': result['metadata']['provider']
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Download all results
                        all_results_json = json.dumps(batch_results, indent=2)
                        st.download_button(
                            "📥 Download All Results (JSON)",
                            all_results_json,
                            f"batch_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                        
                        # Store in session state
                        st.session_state.annotations.extend(batch_results)

if __name__ == "__main__":
    main()