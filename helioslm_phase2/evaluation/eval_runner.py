"""Automated evaluation runner for training checkpoints."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import torch


class EvaluationRunner:
    """
    Run full evaluation suite on model checkpoints.

    Usage:
        runner = EvaluationRunner(model, tokenizer, config)
        results = runner.run_all("checkpoints/best")
        runner.save_results(results, "eval_results.json")
    """

    def __init__(
        self,
        model,
        tokenizer,
        eval_datasets: Dict[str, str],
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_datasets = eval_datasets
        self.device = device

        self.evaluators = {}
        self._init_evaluators()

    def _init_evaluators(self):
        """Initialize evaluators."""
        from .benchmarks import (
            PerplexityEvaluator,
            HellaSwagEvaluator,
            MMLUEvaluator,
            GSM8KEvaluator,
            HumanEvalEvaluator,
            BBHEvaluator,
        )

        if "perplexity" in self.eval_datasets:
            self.evaluators["perplexity"] = PerplexityEvaluator(self.model, self.tokenizer, self.device)
        if "hellaswag" in self.eval_datasets:
            self.evaluators["hellaswag"] = HellaSwagEvaluator(self.model, self.tokenizer, self.device)
        if "mmlu" in self.eval_datasets:
            self.evaluators["mmlu"] = MMLUEvaluator(self.model, self.tokenizer, self.device)
        if "gsm8k" in self.eval_datasets:
            self.evaluators["gsm8k"] = GSM8KEvaluator(self.model, self.tokenizer, self.device)
        if "humaneval" in self.eval_datasets:
            self.evaluators["humaneval"] = HumanEvalEvaluator(self.model, self.tokenizer, self.device)
        if "bbh" in self.eval_datasets:
            self.evaluators["bbh"] = BBHEvaluator(self.model, self.tokenizer, self.device)

    def run_all(self) -> Dict:
        """Run all configured evaluations."""
        results = {}

        for name, evaluator in self.evaluators.items():
            dataset_path = self.eval_datasets[name]
            print(f"🔍 Evaluating {name}...")

            try:
                result = evaluator.evaluate(dataset_path)
                results[name] = result
                print(f"✅ {name}: {result}")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = {"error": str(e)}

        return results

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_path}")

    def compare_checkpoints(self, checkpoint_results: List[Dict]) -> Dict:
        """Compare results across multiple checkpoints."""
        comparison = {}

        for metric in checkpoint_results[0].keys():
            if isinstance(checkpoint_results[0][metric], dict) and "accuracy" in checkpoint_results[0][metric]:
                comparison[metric] = {
                    f"checkpoint_{i}": r[metric].get("accuracy", 0)
                    for i, r in enumerate(checkpoint_results)
                }

        return comparison
