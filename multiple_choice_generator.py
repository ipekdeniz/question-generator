import os
import json
import re
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.core import PromptTemplate

# Ollama model configuration
model_name = "mistral:7b-instruct"  # Change this to your installed Mistral model
llm = Ollama(model=model_name, request_timeout=60.0)
Settings.llm = llm

# Multiple choice question generation template
question_gen_template_str = (
    "I want you to create {num_questions} multiple-choice questions based on the document text provided below. "
    "Please create exactly {num_questions} questions.\n\n"
    "Document text:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "For each question:\n"
    "1. Create a meaningful question related to the text content\n"
    "2. Write 4 options (A, B, C, D) - only one should be the correct answer\n"
    "3. Indicate which option is the correct answer\n"
    "4. Write a brief explanation for the correct answer\n\n"
    "Answer in the following format:\n\n"
    "1. Question: [Question text]\n"
    "A) [Option A]\n"
    "B) [Option B]\n"
    "C) [Option C]\n"
    "D) [Option D]\n"
    "Correct Answer: [Correct option letter]\n"
    "Explanation: [Explanation for the correct answer]\n\n"
    "2. Question: [Question text]\n"
    "A) [Option A]\n"
    "...\n\n"
    "(Please create exactly {num_questions} questions in this format. Number each question clearly.)"
)
question_gen_template = PromptTemplate(question_gen_template_str)

def load_documents(docs_dir="documents"):
    """Load documents from the specified directory"""
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Created '{docs_dir}' directory. Please add your documents there.")
        return None
    
    documents = SimpleDirectoryReader(docs_dir).load_data()
    if not documents:
        print(f"No documents found in '{docs_dir}' directory.")
        return None
    
    print(f"Successfully loaded {len(documents)} documents.")
    return documents

def parse_questions_from_text(text, expected_count=None):
    """Parse questions from LLM response text"""
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

def generate_questions(documents, num_questions=5):
    """Generate multiple-choice questions from documents"""
    if not documents:
        return None
    
    print("Generating questions...")
    
    # Combine document content (limit if too long)
    all_text = "\n\n".join([doc.text for doc in documents])
    max_length = 16000  # Limit to avoid context window overflow
    
    if len(all_text) > max_length:
        print(f"Text too long, using only first {max_length//1000}K characters for question generation.")
        all_text = all_text[:max_length]
    
    # Use LLM to generate questions
    print(f"Generating {num_questions} questions...")
    
    # Send prompt with parameters to LLM
    response = llm.complete(
        question_gen_template.format(
            context=all_text,
            num_questions=num_questions
        )
    )
    
    # Parse LLM response
    response_text = response.text
    print("Processing LLM response...")
    
    # Show first few lines of LLM response for debugging
    first_lines = "\n".join(response_text.split("\n")[:6])
    print(f"First few lines of LLM response: \n{first_lines}\n...")
    
    # Parse response and return questions
    questions = parse_questions_from_text(response_text, expected_count=num_questions)
    
    if not questions:
        print("Failed to parse questions.")
        print("LLM response:", response_text)
        return None
    
    print(f"Successfully parsed {len(questions)} questions.")
    return questions

def save_questions_to_file(questions, output_file="generated_questions.json"):
    """Save generated questions to a file"""
    if questions:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        print(f"Questions saved to '{output_file}'.")

def main():
    print("Document-Based Multiple-Choice Question Generator")
    print("------------------------------------------------")
    
    # Load documents
    documents = load_documents()
    if not documents:
        return
    
    # Get number of questions from user
    while True:
        try:
            num_questions = int(input("How many questions would you like to generate? (1-20): "))
            if 1 <= num_questions <= 20:
                break
            else:
                print("Please enter a number between 1 and 20.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Generate questions
    questions = generate_questions(documents, num_questions)
    
    # Display questions and save to file
    if questions:
        print("\nGenerated Questions:")
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}: {q['question']}")
            for option in q['options']:
                print(option)
            print(f"Correct Answer: {q['correct_answer']}")
            if 'explanation' in q and q['explanation']:
                print(f"Explanation: {q['explanation']}")
        
        save_questions_to_file(questions)
    else:
        print("Failed to generate questions.")

if __name__ == "__main__":
    main() 