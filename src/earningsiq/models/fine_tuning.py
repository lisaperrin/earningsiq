from pathlib import Path
from typing import Optional, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
import pandas as pd
from ..config import settings
from ..utils.logger import log


class FinancialDatasetPreparator:
    def __init__(self):
        pass

    def load_finqa_dataset(self) -> Dataset:
        log.info("Loading financial Q&A dataset")
        dataset_options = [
            ("ChanceFocus/finqa", "train"),
            ("virattt/financial-qa-10K", "train"),
            ("gbharti/finance-alpaca", "train"),
        ]

        for dataset_name, split in dataset_options:
            try:
                log.info(f"Trying dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split=split)
                log.info(f"Successfully loaded {dataset_name} with {len(dataset)} samples")
                return dataset
            except Exception as e:
                log.warning(f"Failed to load {dataset_name}: {e}")
                continue

        # Fallback: create a minimal financial Q&A dataset
        log.warning("Using fallback synthetic financial Q&A dataset")
        return self._create_fallback_dataset()

    def _create_fallback_dataset(self) -> Dataset:
        """Create a small synthetic financial Q&A dataset as fallback."""
        samples = [
            {"context": "Apple Inc. reported revenue of $394.3 billion for fiscal year 2024, an increase of 2% from the prior year. iPhone revenue was $200.6 billion.", "question": "What was Apple's total revenue?", "answer": "Apple's total revenue was $394.3 billion for fiscal year 2024."},
            {"context": "NVIDIA reported record quarterly revenue of $22.1 billion, up 122% from a year ago. Data Center revenue was $18.4 billion, up 217%.", "question": "How much did NVIDIA's revenue grow?", "answer": "NVIDIA's revenue grew 122% year over year to $22.1 billion."},
            {"context": "Microsoft's Intelligent Cloud segment generated $24.3 billion in revenue, with Azure growing 29%. Operating income increased 23%.", "question": "What was Azure's growth rate?", "answer": "Azure grew 29% in the quarter."},
            {"context": "Tesla delivered 484,507 vehicles in Q3 2024. Automotive gross margin was 17.1%. Energy storage deployments reached 6.9 GWh.", "question": "How many vehicles did Tesla deliver?", "answer": "Tesla delivered 484,507 vehicles in Q3 2024."},
            {"context": "Amazon Web Services revenue increased 19% to $24.2 billion. Operating income for AWS was $7.2 billion, representing a 30% margin.", "question": "What was AWS's operating margin?", "answer": "AWS had an operating margin of 30% with $7.2 billion operating income."},
        ] * 20
        return Dataset.from_list(samples)

    def prepare_training_data(self, dataset: Dataset, max_samples: Optional[int] = None) -> Dataset:
        def format_instruction(example):
            context = example.get("pre_text", "") or example.get("context", "") or example.get("input", "") or ""
            question = example.get("question", "") or example.get("instruction", "") or ""
            answer = example.get("answer", "") or example.get("output", "") or example.get("response", "") or ""

            prompt = f"""### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""

            return {"text": prompt}

        formatted_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

        if max_samples:
            formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))

        return formatted_dataset

    def create_custom_dataset(self, documents_df: pd.DataFrame, qa_pairs: list[Dict]) -> Dataset:
        formatted_data = []

        for qa in qa_pairs:
            ticker = qa.get("ticker")
            question = qa.get("question")
            answer = qa.get("answer")

            context_docs = documents_df[documents_df["ticker"] == ticker]["text"].head(3).tolist()
            context = "\n\n".join(context_docs)

            formatted_data.append({
                "text": f"""### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""
            })

        return Dataset.from_list(formatted_data)


class LoRAFineTuner:
    def __init__(
        self,
        model_name: str = None,
        output_dir: Path = None,
        use_4bit: bool = True
    ):
        self.model_name = model_name or settings.llm_model
        self.output_dir = output_dir or settings.models_dir / "finetuned"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_4bit = use_4bit

        self.tokenizer = None
        self.model = None
        self.peft_config = None

    def _is_bitsandbytes_available(self) -> bool:
        """Check if bitsandbytes is installed and working."""
        try:
            import bitsandbytes
            _ = bitsandbytes.__version__
            return True
        except (ImportError, RuntimeError, Exception):
            return False

    def setup_model(self):
        log.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        use_4bit = self.use_4bit and self._is_bitsandbytes_available()

        if self.use_4bit and not use_4bit:
            log.warning("bitsandbytes not installed - using standard precision instead")
            log.warning("To enable 4-bit: pip install bitsandbytes-windows")

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # Load without quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )

        self.model = get_peft_model(self.model, self.peft_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

    def tokenize_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ):
        if self.model is None:
            self.setup_model()

        train_tokenized = self.tokenize_dataset(train_dataset)
        eval_tokenized = self.tokenize_dataset(eval_dataset) if eval_dataset else None

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_steps=100 if eval_tokenized else None,
            eval_strategy="steps" if eval_tokenized else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_tokenized else False,
            report_to="none",
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
        )

        log.info("Starting training")
        trainer.train()

        log.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))

    def load_finetuned_model(self, checkpoint_path: Path = None):
        checkpoint_path = checkpoint_path or self.output_dir

        log.info(f"Loading fine-tuned model from {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                device_map="auto",
            )

    def generate(self, prompt: str, max_length: int = 512) -> str:
        if self.model is None or self.tokenizer is None:
            log.error("Model not loaded")
            return ""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class ModelComparator:
    def __init__(self, base_model, finetuned_model):
        self.base_model = base_model
        self.finetuned_model = finetuned_model

    def compare_on_dataset(self, test_dataset: Dataset) -> Dict:
        base_results = []
        finetuned_results = []

        for example in test_dataset:
            prompt = example["text"].split("### Answer:")[0] + "### Answer:"

            base_response = self.base_model.generate(prompt)
            finetuned_response = self.finetuned_model.generate(prompt)

            base_results.append(base_response)
            finetuned_results.append(finetuned_response)

        return {
            "base_responses": base_results,
            "finetuned_responses": finetuned_results,
            "test_data": test_dataset
        }
