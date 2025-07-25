from src.models.engine import LLMEngine
from src.models.parser import parse_questions_from_text
from src.utils.document_loader import load_documents, split_documents_into_nodes
from src.api.schemas import Question, QuestionResponse
import uuid

class QuestionGenerationService:
    def __init__(self, engine: LLMEngine = None):
        self.engine = engine or LLMEngine()

    def generate_questions(self, num_questions: int) -> QuestionResponse:
        documents = load_documents()
        if not documents:
            return None
        if not self.engine.index:
            self.engine.load_index()
        if not self.engine.index:
            nodes = split_documents_into_nodes(documents)
            self.engine.create_index(nodes)
        raw_questions = self.engine.generate_questions(
            num_questions=num_questions,
            similarity_top_k=10
        )
        # None check ekle
        if raw_questions is None:
            print("ERROR: LLM returned None for question generation")
            return None
        if not isinstance(raw_questions, str):
            print(f"ERROR: LLM returned non-string type: {type(raw_questions)}")
            return None
        parsed_questions = parse_questions_from_text(raw_questions, num_questions)
        questions = []
        for q in parsed_questions:
            options = []
            for opt in q['options']:
                label = opt[0]
                text = opt[3:].strip()
                options.append({"label": label, "text": text})
            questions.append(Question(
                id=str(uuid.uuid4()),
                question=q['question'],
                options=options,
                correct_answer=q['correct_answer'],
                explanation=q['explanation']
            ))
        return QuestionResponse(questions=questions, count=len(questions)) 