import json

def quick_validation_check(json_data):
    """
    Quick validation check for your specific JSON data
    """
    text = json_data['text']
    entities = json_data['entities']
    
    errors = []
    
    for i, entity in enumerate(entities):
        start = entity['start_char']
        end = entity['end_char']
        expected = entity['text']
        
        if start < 0 or end > len(text) or start >= end:
            errors.append(f"Entity {i}: Invalid position [{start}:{end}]")
            continue
            
        actual = text[start:end]
        
        if actual != expected:
            errors.append(f"Entity {i}: Expected '{expected}', got '{actual}' at [{start}:{end}]")
    
    return errors

# Test with your data
import json

with open("annotations_corrected.json", "r", encoding='utf-8') as file:
    json_data = json.load(file)

errors = quick_validation_check(json_data)
if errors:
    for error in errors:
        print(error)
else:
    print("All annotations appear to be correct!")