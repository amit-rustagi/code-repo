import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import DataCollatorForLanguageModeling
import wandb

class MistralFineTuner:
    def __init__(
        self, 
        model_name='mistralai/Mistral-7B-v0.1',
        dataset_name='your_dataset_here',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1
    ):
        """
        Initialize Mistral fine-tuning configuration
        
        Args:
            model_name (str): Hugging Face model identifier
            dataset_name (str): Dataset for fine-tuning
            lora_r (int): LoRA rank
            lora_alpha (int): LoRA alpha scaling
            lora_dropout (float): LoRA dropout rate
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model in 4-bit quantization for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            load_in_4bit=True
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", 
                "k_proj", 
                "v_proj", 
                "o_proj", 
                "gate_proj", 
                "up_proj", 
                "down_proj"
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Load dataset
        self.dataset = self._load_and_preprocess_dataset(dataset_name)
    
    def _load_and_preprocess_dataset(self, dataset_name):
        """
        Load and preprocess training dataset
        
        Args:
            dataset_name (str): Hugging Face dataset identifier
        
        Returns:
            Processed dataset
        """
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                max_length=512, 
                padding='max_length'
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=dataset['train'].column_names
        )
        
        return tokenized_dataset
    
    def prepare_training(
        self, 
        output_dir='./mistral_finetune',
        learning_rate=5e-5,
        batch_size=4,
        epochs=3
    ):
        """
        Prepare training arguments and trainer
        
        Args:
            output_dir (str): Directory to save model
            learning_rate (float): Training learning rate
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
        
        Returns:
            Trainer instance
        """
        # Initialize W&B for experiment tracking
        wandb.init(project='mistral-finetune')
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_dir='./logs',
            logging_steps=10,
            push_to_hub=False,
            fp16=True
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            data_collator=data_collator
        )
        
        return trainer
    
    def train(self):
        """
        Execute model fine-tuning
        """
        trainer = self.prepare_training()
        trainer.train()
        
        # Save fine-tuned model
        self.model.save_pretrained('./fine_tuned_mistral')
        self.tokenizer.save_pretrained('./fine_tuned_mistral')
    
    def inference(self, prompt, max_length=200):
        """
        Generate inference using fine-tuned model
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum generation length
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length, 
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0])

def main():
    # Initialize fine-tuner
    fine_tuner = MistralFineTuner(
        dataset_name='your_dataset_here'
    )
    
    # Train model
    fine_tuner.train()
    
    # Example inference
    result = fine_tuner.inference("Your test prompt here")
    print(result)

if __name__ == "__main__":
    main()
