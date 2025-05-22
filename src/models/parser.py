"""
Question parsing utilities for extracting structured data from LLM outputs.
"""
import re
from typing import List, Dict, Any, Optional


def parse_questions_from_text(text: str, expected_count: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Parse questions from LLM response text.
    
    Args:
        text: LLM response text containing questions
        expected_count: Expected number of questions (for fallback strategies)
        
    Returns:
        List of parsed question dictionaries
    """
    questions = []
    
    # Try to find all questions in the text
    pattern1 = r'(?:\d+\.|#\d+|Question\s+\d+:)\s*(.*?)(?=(?:\d+\.|#\d+|Question\s+\d+:)|$)'
    pattern2 = r'(?:\d+\.\s*Question:?)\s*(.*?)(?=(?:\d+\.\s*Question:?)|$)'
    
    # Try with first pattern
    question_blocks = re.findall(pattern1, text, re.DOTALL)
    
    # If not found, try with second pattern
    if not question_blocks:
        question_blocks = re.findall(pattern2, text, re.DOTALL)
    
    # If still not found, try manual splitting
    if not question_blocks and expected_count:
        print("No pattern matched, trying manual text splitting...")
        # Split text into approximately equal parts
        avg_length = len(text) // expected_count
        question_blocks = []
        for i in range(expected_count):
            start_idx = max(0, i * avg_length - 100)  # Start a bit earlier
            end_idx = min(len(text), (i + 1) * avg_length + 100)  # End a bit later
            question_blocks.append(text[start_idx:end_idx])
    
    print(f"Found {len(question_blocks)} potential question blocks for parsing.")
    
    for i, block in enumerate(question_blocks):
        try:
            # Find question text
            question_text = ""
            
            # Try to find the question sentence
            q_pattern = r'(?:Question)?:?\s*(.*?)(?=\s*(?:A[\.\)]|A\s+\)))'
            q_match = re.search(q_pattern, block, re.DOTALL | re.IGNORECASE)
            
            if q_match:
                question_text = q_match.group(1).strip()
            else:
                # Alternative: use first line as question
                first_line = block.split('\n')[0].strip()
                if first_line and not first_line.startswith(('A)', 'B)', 'C)', 'D)')):
                    question_text = first_line
            
            # Clean question text
            for prefix in ["question:", "1.", "2.", "3.", "4.", "5."]:
                if question_text.lower().startswith(prefix):
                    question_text = question_text[len(prefix):].strip()
            
            if not question_text:
                print(f"No question text found in block {i+1}")
                continue
            
            # Find options (both A) and A. format)
            options = []
            option_patterns = [
                r'([A-D])[\.\)]\s+(.*?)(?=(?:[A-D][\.\)])|(?:Correct\s+Answer):|$)',
                r'([A-D])\s*[\.\)]\s+(.*?)(?=(?:[A-D]\s*[\.\)])|(?:Correct\s+Answer):|$)'
            ]
            
            for pattern in option_patterns:
                options_matches = re.findall(pattern, block, re.DOTALL | re.IGNORECASE)
                if options_matches:
                    for letter, text in options_matches:
                        options.append(f"{letter}) {text.strip()}")
                    break  # If one pattern matches, don't try others
            
            # Find correct answer (various formats)
            correct_patterns = [
                r'(?:Correct\s+Answer):\s*([A-D])',
                r'(?:Answer):\s*([A-D])',
                r'(?:Correct):\s*([A-D])'
            ]
            
            correct_answer = None
            for pattern in correct_patterns:
                correct_match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
                if correct_match:
                    correct_answer = correct_match.group(1).strip().upper()
                    break
            
            if not correct_answer and options:
                # If no correct answer is specified, use the first option
                correct_answer = options[0][0]
                print(f"No correct answer found for block {i+1}, using default: {correct_answer}")
            
            # Find explanation (various formats)
            explanation_patterns = [
                r'(?:Explanation):\s*(.*?)(?=$)',
                r'(?:Explanation):\s*(.*?)(?=\d+\.\s*Question:|$)'
            ]
            
            explanation = ""
            for pattern in explanation_patterns:
                explanation_match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                    break
            
            # Add question if at least 2 options are found
            if len(options) >= 2 and question_text:
                # Fill in missing options
                while len(options) < 4:
                    missing_letter = 'ABCD'[len(options)]
                    options.append(f"{missing_letter}) [Missing option]")
                
                # Check for duplicates
                if not any(q["question"] == question_text for q in questions):
                    questions.append({
                        "question": question_text,
                        "options": options[:4],  # Take at most 4 options
                        "correct_answer": correct_answer,
                        "explanation": explanation
                    })
        except Exception as e:
            print(f"Error parsing block {i+1}: {e}")
            continue
    
    return questions 