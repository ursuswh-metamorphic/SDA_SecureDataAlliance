"""fedrag: LLM querier for ensemble synthesis in Federated RAG."""

import os
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid deadlocks during tokenization

log = logging.getLogger(__name__)


class LLMQuerier:
    def __init__(self, model_name: str, use_gpu: bool = False):
        """Initialize a causal LLM for answering / ensemble synthesis."""
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # set pad token if empty
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )


    def synthesize_ensemble(
        self,
        ensemble_prompt: str,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Synthesize ensemble answer from multiple client answers using LLM.
        
        Args:
            ensemble_prompt: Pre-formatted prompt with multiple client answers
                             (ví dụ: tổng hợp các câu trả lời / đoạn retrieve từ nhiều client)
            max_new_tokens: Max tokens to generate
            
        Returns:
            Synthesized answer (str)
        """
        try:
            formatted_prompt = self.__format_ensemble_prompt(ensemble_prompt)
            
            inputs = self.tokenizer(
                formatted_prompt,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ).to(self.device)
            
            attention_mask = (inputs.input_ids != self.tokenizer.pad_token_id).long()
            
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                early_stopping=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            ensemble_answer = self.__extract_ensemble_answer(
                generated_text,
                formatted_prompt,
            )
            
            log.info(
                "Ensemble synthesis completed: "
                f"{ensemble_answer[:100] if ensemble_answer else 'None'}"
            )
            return ensemble_answer
        except Exception as e:
            log.error(f"Ensemble synthesis error: {e}")
            raise

    @classmethod
    def __format_ensemble_prompt(cls, ensemble_prompt: str) -> str:
        """Format ensemble prompt for LLM processing specialized for finance."""
        system_instruction = (
            "You are a senior financial analyst specializing in synthesizing answers from multiple "
            "retrieval sources into one accurate and concise final response. "
            "Provide a brief reasoning step (1–2 sentences) before the final answer. "
            "Focus on factual correctness, consistency across sources, and eliminate contradictions. "
            "Maintain a professional analytical tone consistent with financial reporting."
        )
        return f"""{system_instruction}

{ensemble_prompt}

Final synthesized answer:"""

    @classmethod
    def __extract_ensemble_answer(cls, generated_text: str, original_prompt: str) -> str:
        """Extract ensemble answer from generated text."""
        # Remove prompt prefix
        response = generated_text[len(original_prompt):].strip()
        
        # Nếu model sinh nhiều dòng, lấy dòng đầu (thường là câu trả lời chính)
        if not response:
            return None

        first_line = response.split("\n")[0].strip()
        if first_line:
            return first_line

        return response[:200]
