"""
A simple quiz application that uses generated multiple-choice questions
"""
import os
import json
import random

def load_questions(file_path="generated_questions.json"):
    """Load questions from JSON file"""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        print("Please run 'multiple_choice_generator.py' first to create questions.")
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"Successfully loaded {len(questions)} questions.")
        return questions
    except json.JSONDecodeError:
        print(f"Could not load questions from '{file_path}'. Invalid JSON format.")
        return None

def display_question(question, question_num):
    """Display a question on screen"""
    print(f"\nQuestion {question_num}: {question['question']}")
    for option in question['options']:
        print(option)

def get_user_answer():
    """Get answer from user"""
    while True:
        answer = input("\nYour answer (A, B, C, D or 'q' to quit): ").strip().upper()
        if answer in ['A', 'B', 'C', 'D', 'Q']:
            return answer
        print("Please enter a valid answer (A, B, C, D or Q).")

def check_answer(question, user_answer):
    """Check user's answer"""
    if user_answer == question['correct_answer']:
        print("\n✓ Correct answer!")
        print(f"Explanation: {question['explanation']}")
        return True
    else:
        print("\n✗ Wrong answer!")
        print(f"Correct answer: {question['correct_answer']}")
        print(f"Explanation: {question['explanation']}")
        return False

def run_quiz(questions, randomize=True):
    """Run the quiz application"""
    if not questions:
        return
    
    if randomize:
        random.shuffle(questions)
    
    score = 0
    total_questions = len(questions)
    
    for i, q in enumerate(questions, 1):
        display_question(q, i)
        user_answer = get_user_answer()
        
        if user_answer == 'Q':
            print("\nQuiz terminated.")
            break
        
        if check_answer(q, user_answer):
            score += 1
        
        print(f"\nCurrent score: {score}/{i}")
        
        if i < total_questions:
            input("\nPress Enter to continue...")
    
    if user_answer != 'Q':
        print(f"\nQuiz completed! Total score: {score}/{total_questions}")
        percentage = (score / total_questions) * 100
        print(f"Success rate: {percentage:.1f}%")

def main():
    print("Multiple-Choice Quiz from Generated Questions")
    print("--------------------------------------------")
    
    questions = load_questions()
    if questions:
        print(f"\nTotal questions available: {len(questions)}")
        print("Starting quiz...\n")
        
        randomize = input("Would you like to randomize questions? (Y/N): ").strip().upper() == 'Y'
        run_quiz(questions, randomize)
    else:
        print("\nCould not start quiz. Please run 'multiple_choice_generator.py' first to create questions.")

if __name__ == "__main__":
    main() 