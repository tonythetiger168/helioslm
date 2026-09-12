"""Benchmark evaluators for downstream task assessment."""
import json
import re
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class BaseEvaluator:
    """Base class for benchmark evaluators."""

    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, dataset_path: str) -> Dict:
        raise NotImplementedError

    def generate_answer(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Generate answer for a given prompt."""
        self.model.eval()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_ids=input_tensor)
                logits = outputs["logits"]

                if temperature == 0.0:
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                input_tensor = torch.cat([input_tensor, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        generated = self.tokenizer.decode(input_tensor[0].tolist(), skip_special_tokens=True)
        return generated[len(prompt):].strip()


class PerplexityEvaluator(BaseEvaluator):
    """Evaluate perplexity on a validation set."""

    def evaluate(self, dataset_path: str) -> Dict:
        import math

        total_loss = 0
        total_tokens = 0

        with open(dataset_path) as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")

                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                input_ids = torch.tensor([tokens], device=self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.get("loss", 0)
                    if isinstance(loss, torch.Tensor):
                        loss = loss.item()

                total_loss += loss * len(tokens)
                total_tokens += len(tokens)

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)

        return {"perplexity": perplexity, "avg_loss": avg_loss}


class HellaSwagEvaluator(BaseEvaluator):
    """HellaSwag commonsense reasoning benchmark."""

    def evaluate(self, dataset_path: str) -> Dict:
        correct = 0
        total = 0

        with open(dataset_path) as f:
            for line in f:
                data = json.loads(line)
                context = data["ctx"]
                endings = data["endings"]
                label = data["label"]

                scores = []
                for ending in endings:
                    text = context + " " + ending
                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    input_ids = torch.tensor([tokens], device=self.device)

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, labels=input_ids)
                        loss = outputs.get("loss", 0)
                        if isinstance(loss, torch.Tensor):
                            loss = loss.item()

                    scores.append(-loss)  # Lower loss = higher score

                predicted = scores.index(max(scores))
                if predicted == label:
                    correct += 1
                total += 1

        return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


class MMLUEvaluator(BaseEvaluator):
    """MMLU (Massive Multitask Language Understanding) benchmark."""

    def evaluate(self, dataset_path: str) -> Dict:
        correct = 0
        total = 0
        category_correct: Dict[str, int] = {}
        category_total: Dict[str, int] = {}

        with open(dataset_path) as f:
            for line in f:
                data = json.loads(line)
                question = data["question"]
                choices = data["choices"]
                answer = data["answer"]
                category = data.get("category", "unknown")

                prompt = f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Answer:"

                generated = self.generate_answer(prompt, max_tokens=10, temperature=0.0)

                # Extract answer letter
                match = re.search(r"[A-D]", generated.upper())
                predicted = match.group(0) if match else "A"

                correct_answer = chr(65 + answer)
                is_correct = (predicted == correct_answer)

                if is_correct:
                    correct += 1
                    category_correct[category] = category_correct.get(category, 0) + 1

                total += 1
                category_total[category] = category_total.get(category, 0) + 1

        category_acc = {
            cat: category_correct.get(cat, 0) / category_total[cat]
            for cat in category_total
        }

        return {
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
            "category_accuracy": category_acc,
        }


class GSM8KEvaluator(BaseEvaluator):
    """GSM8K math word problems benchmark."""

    def evaluate(self, dataset_path: str) -> Dict:
        correct = 0
        total = 0

        with open(dataset_path) as f:
            for line in f:
                data = json.loads(line)
                question = data["question"]
                answer = data["answer"]

                # Extract numerical answer
                true_answer = self._extract_number(answer)

                prompt = f"Solve this math problem step by step:\n{question}\nAnswer:"
                generated = self.generate_answer(prompt, max_tokens=256, temperature=0.0)

                predicted_answer = self._extract_number(generated)

                if abs(predicted_answer - true_answer) < 1e-3:
                    correct += 1
                total += 1

        return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}

    def _extract_number(self, text: str) -> float:
        """Extract the last number from text."""
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            return float(numbers[-1])
        return 0.0


class HumanEvalEvaluator(BaseEvaluator):
    """HumanEval code generation benchmark."""

    def evaluate(self, dataset_path: str) -> Dict:
        results = []

        with open(dataset_path) as f:
            for line in f:
                data = json.loads(line)
                prompt = data["prompt"]
                test = data["test"]
                entry_point = data["entry_point"]

                generated = self.generate_answer(prompt, max_tokens=512, temperature=0.2)

                # Simple syntax check
                try:
                    compile(generated, "<string>", "exec")
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False

                results.append({
                    "task_id": data.get("task_id", ""),
                    "syntax_valid": syntax_valid,
                    "generated": generated,
                })

        valid = sum(1 for r in results if r["syntax_valid"])
        return {
            "syntax_valid_rate": valid / max(len(results), 1),
            "total": len(results),
            "results": results,
        }


class BBHEvaluator(BaseEvaluator):
    """Big Bench Hard (BBH) benchmark."""

    def evaluate(self, dataset_path: str) -> Dict:
        # Placeholder: BBH requires specific prompting strategies
        return {"status": "not_implemented", "note": "BBH evaluation requires chain-of-thought prompting"}
