import json
import random
import os
import sys
import logging
from typing import List, Dict

# Thêm đường dẫn gốc vào sys.path để import các module local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llms.llm import get_llm
from config import Config
from llama_index.core import Document
from llama_index.core.evaluation import DatasetGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_medical_test_set(corpus_path: str, output_path: str, num_questions: int = 50):
    """
    Tạo tập dữ liệu kiểm thử tổng hợp từ rag_corpus.json.
    """
    logger.info(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # Lấy ngẫu nhiên các đoạn văn để tạo câu hỏi
    sampled_nodes = random.sample(corpus, min(num_questions, len(corpus)))
    
    # Khởi tạo LLM từ config
    cfg = Config()
    llm_name = getattr(cfg, "llm", "nvidia")
    logger.info(f"Using LLM: {llm_name} to generate questions...")
    llm = get_llm(llm_name)

    test_data = []

    for i, node in enumerate(sampled_nodes):
        logger.info(f"[{i+1}/{num_questions}] Generating question for node: {node['id']}")
        
        # Tạo prompt thủ công để đảm bảo định dạng chính xác
        prompt = f"""
Given the following medical context, generate a question and its corresponding answer.
The question should be answerable ONLY using the provided context.
Provide the output in JSON format with 'question' and 'answer' keys.

Context:
{node['text']}

JSON Output:
"""
        try:
            response = llm.complete(prompt)
            # Giả sử LLM trả về JSON sạch hoặc chúng ta cần trích xuất nó
            res_text = response.text.strip()
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            elif "```" in res_text:
                res_text = res_text.split("```")[1].split("```")[0].strip()
            
            # Làm sạch các ký tự lạ nếy có
            start_idx = res_text.find('{')
            end_idx = res_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                res_text = res_text[start_idx:end_idx+1]

            content = json.loads(res_text)
            
            # Format theo cấu trúc của data_100.json
            entry = {
                "key_content": {
                    "reference": [node['text']],
                    "reference_idx": [node['id']], # Sử dụng ID chuỗi từ rag_corpus.json
                    "question": content['question'],
                    "answer": content['answer']
                },
                "other_info": {
                    "doc_id": node['id'],
                    "title": node.get('title', ''),
                    "domain": "medical"
                }
            }
            test_data.append(entry)
        except Exception as e:
            logger.error(f"Failed to generate for node {node['id']}: {e}")

    logger.info(f"Saving {len(test_data)} test cases to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    
    logger.info("Generation complete.")

if __name__ == "__main__":
    corpus_file = os.path.join(os.path.dirname(__file__), "rag_corpus.json")
    output_file = os.path.join(os.path.dirname(__file__), "medical_test_50.json")
    
    # Chạy tạo 50 câu hỏi
    generate_medical_test_set(corpus_file, output_file, num_questions=50)
