"""
Question parsing utilities for extracting structured data from LLM outputs.
"""
import re
from typing import List, Dict, Any, Optional


def _extract_question_blocks(text: str, expected_count: Optional[int] = None) -> List[str]:
    """Extract individual question blocks from LLM response text."""
    # Try to find all questions in the text
    pattern1 = r'(?:\d+\.|#\d+|Question\s+\d+:)\s*(.*?)(?=(?:\d+\.|#\d+|Question\s+\d+:)|$)' # "Question 1:", "Question 2:" vs
    pattern2 = r'(?:\d+\.\s*Question:?)\s*(.*?)(?=(?:\d+\.\s*Question:?)|$)' # "1. Question:", "1. Question", "2. Question:" vs
    
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
    
    return question_blocks


def _extract_question_text(block: str) -> str:
    """Extract question text from a question block."""
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
    
    return question_text


def _extract_options(block: str) -> List[str]:
    """Extract answer options from a question block."""
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
    
    return options


def _extract_correct_answer(block: str, options: List[str]) -> str:
    """Extract correct answer from a question block."""
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
    
    return correct_answer


def _extract_explanation(block: str) -> str:
    """Extract explanation from a question block."""
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
    
    return explanation


def _validate_and_complete_options(options: List[str]) -> List[str]:
    """Validate and complete missing options."""
    # Fill in missing options
    while len(options) < 4:
        missing_letter = 'ABCD'[len(options)]
        options.append(f"{missing_letter}) [Missing option]")
    
    return options[:4]  # Take at most 4 options


def _parse_single_question(block: str, block_index: int) -> Optional[Dict[str, Any]]:
    """Parse a single question block into a structured question."""
    try:
        question_text = _extract_question_text(block)
        if not question_text:
            print(f"No question text found in block {block_index + 1}")
            return None
        
        options = _extract_options(block)
        if len(options) < 2:
            print(f"Not enough options found in block {block_index + 1}")
            return None
        
        correct_answer = _extract_correct_answer(block, options)
        if not correct_answer:
            print(f"No correct answer found for block {block_index + 1}, using default: A")
            correct_answer = "A"
        
        explanation = _extract_explanation(block)
        options = _validate_and_complete_options(options)
        
        return {
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation
        }
    
    except Exception as e:
        print(f"Error parsing block {block_index + 1}: {e}")
        return None


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
    
    # Extract question blocks
    question_blocks = _extract_question_blocks(text, expected_count)
    print(f"Found {len(question_blocks)} potential question blocks for parsing.")
    
    # Parse each block
    for i, block in enumerate(question_blocks):
        parsed_question = _parse_single_question(block, i)
        if parsed_question:
            # Check for duplicates
            if not any(q["question"] == parsed_question["question"] for q in questions):
                questions.append(parsed_question)
    
    return questions 