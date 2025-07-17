import json

def validate_annotations(json_file_path):
    """
    Validate that start_char and end_char positions in annotations match the actual text.
    
    Args:
        json_file_path (str): Path to the JSON file with annotations
    
    Returns:
        dict: Validation results with errors and statistics
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data['text']
    entities = data['entities']
    
    validation_results = {
        'total_entities': len(entities),
        'correct_entities': 0,
        'errors': [],
        'warnings': []
    }
    
    print(f"Validating {len(entities)} annotations...")
    print("=" * 50)
    
    for i, entity in enumerate(entities):
        start_char = entity['start_char']
        end_char = entity['end_char']
        expected_text = entity['text']
        
        # Extract actual text from the source using the character positions
        try:
            actual_text = text[start_char:end_char]
            
            # Check if texts match exactly
            if actual_text == expected_text:
                validation_results['correct_entities'] += 1
                print(f"✓ Entity {i+1}: '{expected_text}' - CORRECT")
            else:
                error_info = {
                    'entity_index': i,
                    'expected_text': expected_text,
                    'actual_text': actual_text,
                    'start_char': start_char,
                    'end_char': end_char,
                    'label': entity['label']
                }
                validation_results['errors'].append(error_info)
                print(f"✗ Entity {i+1}: MISMATCH")
                print(f"  Expected: '{expected_text}'")
                print(f"  Actual:   '{actual_text}'")
                print(f"  Position: [{start_char}:{end_char}]")
                print(f"  Label: {entity['label']}")
                
                # Show context around the error
                context_start = max(0, start_char - 20)
                context_end = min(len(text), end_char + 20)
                context = text[context_start:context_end]
                print(f"  Context: ...{context}...")
                print()
                
        except IndexError:
            error_info = {
                'entity_index': i,
                'expected_text': expected_text,
                'start_char': start_char,
                'end_char': end_char,
                'error': 'Index out of range'
            }
            validation_results['errors'].append(error_info)
            print(f"✗ Entity {i+1}: INDEX OUT OF RANGE")
            print(f"  Text: '{expected_text}'")
            print(f"  Position: [{start_char}:{end_char}]")
            print(f"  Text length: {len(text)}")
            print()
    
    # Additional checks
    print("\nAdditional Validation Checks:")
    print("=" * 30)
    
    # Check for overlapping annotations
    sorted_entities = sorted(entities, key=lambda x: x['start_char'])
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
            print(f"⚠ Overlap detected:")
            print(f"  '{current['text']}' [{current['start_char']}:{current['end_char']}]")
            print(f"  '{next_entity['text']}' [{next_entity['start_char']}:{next_entity['end_char']}]")
    
    # Check for zero-length annotations
    zero_length = [e for e in entities if e['start_char'] == e['end_char']]
    if zero_length:
        validation_results['warnings'].extend(zero_length)
        print(f"⚠ Found {len(zero_length)} zero-length annotations")
    
    # Summary
    print(f"\nValidation Summary:")
    print("=" * 20)
    print(f"Total entities: {validation_results['total_entities']}")
    print(f"Correct entities: {validation_results['correct_entities']}")
    print(f"Errors: {len(validation_results['errors'])}")
    print(f"Warnings: {len(validation_results['warnings'])}")
    print(f"Accuracy: {validation_results['correct_entities']/validation_results['total_entities']*100:.1f}%")
    
    return validation_results

def fix_annotation_positions(json_file_path, output_file_path=None, strategy='closest'):
    """
    Automatically fix annotation positions by searching for the text.
    
    Args:
        json_file_path (str): Path to the JSON file with annotations
        output_file_path (str): Path to save corrected annotations (optional)
        strategy (str): Strategy for handling multiple matches ('closest', 'first', 'interactive')
    
    Returns:
        dict: Corrected data with statistics
    """
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data['text']
    entities = data['entities']
    
    fixed_entities = []
    stats = {
        'total': len(entities),
        'already_correct': 0,
        'fixed': 0,
        'unfixable': 0,
        'multiple_matches': 0
    }
    
    print(f"Attempting to fix {len(entities)} annotations...")
    print("=" * 50)
    
    for i, entity in enumerate(entities):
        expected_text = entity['text']
        start_char = entity['start_char']
        end_char = entity['end_char']
        
        # Check if current position is correct
        try:
            if start_char >= 0 and end_char <= len(text) and text[start_char:end_char] == expected_text:
                fixed_entities.append(entity)
                stats['already_correct'] += 1
                print(f"✓ Entity {i+1}: '{expected_text}' - Already correct")
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
                print(f"✓ Entity {i+1}: '{expected_text}' - Fixed with fuzzy matching [{fixed_pos[0]}:{fixed_pos[1]}]")
            else:
                # Text not found, keep original but mark as unfixable
                fixed_entities.append(entity)
                stats['unfixable'] += 1
                print(f"✗ Entity {i+1}: '{expected_text}' - Could not fix (not found)")
        elif len(found_positions) == 1:
            # Only one match found, use it
            new_start, new_end = found_positions[0]
            entity_copy = entity.copy()
            entity_copy['start_char'] = new_start
            entity_copy['end_char'] = new_end
            fixed_entities.append(entity_copy)
            stats['fixed'] += 1
            print(f"✓ Entity {i+1}: '{expected_text}' - Fixed [{new_start}:{new_end}]")
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
                print(f"✓ Entity {i+1}: '{expected_text}' - Fixed [{closest_pos[0]}:{closest_pos[1]}] (closest of {len(found_positions)} matches)")
            
            elif strategy == 'first':
                # Use the first occurrence
                first_pos = found_positions[0]
                entity_copy = entity.copy()
                entity_copy['start_char'] = first_pos[0]
                entity_copy['end_char'] = first_pos[1]
                fixed_entities.append(entity_copy)
                stats['fixed'] += 1
                print(f"✓ Entity {i+1}: '{expected_text}' - Fixed [{first_pos[0]}:{first_pos[1]}] (first of {len(found_positions)} matches)")
            
            elif strategy == 'interactive':
                # Let user choose
                print(f"? Entity {i+1}: '{expected_text}' - Found {len(found_positions)} matches:")
                for j, pos in enumerate(found_positions):
                    context = get_context(text, pos[0], pos[1])
                    print(f"  {j+1}. Position [{pos[0]}:{pos[1]}] - Context: ...{context}...")
                
                while True:
                    try:
                        choice = input(f"Choose position (1-{len(found_positions)}, or 's' to skip): ").strip()
                        if choice.lower() == 's':
                            fixed_entities.append(entity)
                            break
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(found_positions):
                            chosen_pos = found_positions[choice_idx]
                            entity_copy = entity.copy()
                            entity_copy['start_char'] = chosen_pos[0]
                            entity_copy['end_char'] = chosen_pos[1]
                            fixed_entities.append(entity_copy)
                            stats['fixed'] += 1
                            print(f"✓ Entity {i+1}: '{expected_text}' - Fixed [{chosen_pos[0]}:{chosen_pos[1]}]")
                            break
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number or 's'.")
    
    # Save corrected data
    corrected_data = data.copy()
    corrected_data['entities'] = fixed_entities
    
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, indent=2, ensure_ascii=False)
        print(f"\nCorrected annotations saved to: {output_file_path}")
    
    # Print statistics
    print(f"\nFix Statistics:")
    print("=" * 20)
    print(f"Total entities: {stats['total']}")
    print(f"Already correct: {stats['already_correct']}")
    print(f"Successfully fixed: {stats['fixed']}")
    print(f"Multiple matches handled: {stats['multiple_matches']}")
    print(f"Unfixable: {stats['unfixable']}")
    print(f"Success rate: {(stats['already_correct'] + stats['fixed'])/stats['total']*100:.1f}%")
    
    return corrected_data, stats

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

def get_context(text, start, end, context_length=20):
    """Get context around a position"""
    context_start = max(0, start - context_length)
    context_end = min(len(text), end + context_length)
    return text[context_start:context_end]

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    json_file = "annotations_input.json"
    
    # Validate annotations
    results = validate_annotations(json_file)
    
    # If there are errors, attempt to fix them
    if results['errors']:
        print("\nAttempting to fix annotation positions...")
        fixed_data = fix_annotation_positions(json_file, "annotations_fixed.json")
        
        # Validate the fixed annotations
        print("\nValidating fixed annotations...")
        validate_annotations("annotations_fixed.json")