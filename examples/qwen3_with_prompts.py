"""
Example: Training Qwen3 with different system prompts
"""

from brute_force_training import Qwen3Trainer

def train_math_tutor():
    """Example: Math tutoring model"""
    trainer = Qwen3Trainer(
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./qwen3_math_tutor",
        device="cuda"
    )
    
    trainer.train_and_validate(
        dataset_name="your_math_dataset",
        input_column="problem",
        output_column="solution",
        
        # Math-specific system prompt
        system_prompt="You are a mathematics tutor. Provide step-by-step solutions to math problems. Explain your reasoning clearly and show all work.",
        
        max_steps=5000,
        train_batch_size=2,
        learning_rate=1e-5
    )

def train_code_assistant():
    """Example: Code assistance model"""
    trainer = Qwen3Trainer(
        model_name="Qwen/Qwen3-4B-Thinking-2507", 
        output_dir="./qwen3_code_assistant",
        device="cuda"
    )
    
    trainer.train_and_validate(
        dataset_name="your_code_dataset",
        input_column="question", 
        output_column="code",
        
        # Code-specific system prompt
        system_prompt="You are a coding assistant. Write clean, efficient, and well-commented code. Always explain what your code does.",
        
        max_steps=5000,
        train_batch_size=2,
        learning_rate=1e-5
    )

def train_creative_writer():
    """Example: Creative writing model"""
    trainer = Qwen3Trainer(
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./qwen3_creative_writer", 
        device="cuda"
    )
    
    trainer.train_and_validate(
        dataset_name="your_writing_dataset",
        input_column="prompt",
        output_column="story",
        
        # Creative writing system prompt
        system_prompt="You are a creative writing assistant. Write engaging, imaginative stories with vivid descriptions and compelling characters.",
        
        max_steps=5000,
        train_batch_size=2,
        learning_rate=1e-5
    )

def train_no_system_prompt():
    """Example: Training without system prompt (like the old behavior)"""
    trainer = Qwen3Trainer(
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_dir="./qwen3_no_prompt",
        device="cuda"
    )
    
    trainer.train_and_validate(
        dataset_name="your_dataset",
        input_column="input",
        output_column="output",
        
        # No system prompt - just like before
        system_prompt=None,
        
        max_steps=5000,
        train_batch_size=2,
        learning_rate=1e-5
    )

if __name__ == "__main__":
    print("Choose a training example:")
    print("1. Math tutor")
    print("2. Code assistant") 
    print("3. Creative writer")
    print("4. No system prompt")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        train_math_tutor()
    elif choice == "2":
        train_code_assistant()
    elif choice == "3":
        train_creative_writer()
    elif choice == "4":
        train_no_system_prompt()
    else:
        print("Invalid choice!")
