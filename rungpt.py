import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from dataset import TokenizedDataset
from tokenizers.implementations import ByteLevelBPETokenizer

# Path to the dataset directory (full transcripts)
transcripts_dir = "C:/gitrepos/nlp/yt_ascii"



# 1. Define GPT configuration (minimal configuration for small model)
config = GPT2Config(
    vocab_size=32768,  # Tokenizer vocab size
    n_positions=256+128,   # Maximum sequence length
    n_embd=1024,        # Hidden size (smaller hidden size for faster training)
    n_layer=32,         # Fewer layers
    n_head=16,          # Fewer attention heads
    #n_inner=512,       # Inner size of feed-forward layer
    activation_function="gelu_new",  # Activation function
    resid_pdrop=0.075,   # Dropout rate
    embd_pdrop=0.075,    # Dropout on embedding layer
    attn_pdrop=0.075,    # Dropout on attention heads
)

# 2. Initialize GPT-like model (without pre-trained weights)
model = GPT2LMHeadModel(config=config)

    
# 3. Initialize weights from scratch (using Kaiming or Xavier initialization)
# def custom_weight_init(module):
#     if isinstance(module, (nn.Linear, nn.Conv1d)):
#         nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         nn.init.normal_(module.weight, mean=0, std=0.02)  # Initialize embeddings from normal distribution
#     elif isinstance(module, nn.LayerNorm):
#         nn.init.ones_(module.weight)  # Initialize LayerNorm weights to 1
#         nn.init.zeros_(module.bias)   # Initialize LayerNorm bias to 0

# # Apply the custom weight initialization
# model.apply(custom_weight_init)
#model.transformer.load_state_dict(torch.load("C:/data/nlp/xfmer.pt",weights_only=True))
#model.lm_head.load_state_dict(torch.load("C:/data/nlp/lmhead.pt",weights_only=True))

model   = model.from_pretrained("C:/gitrepos/nlp/scratch_gpt2_youtube_script/checkpoint-128")
# 4. Tokenizer setup (using GPT-2 tokenizer)
tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename="stein_tokenizer_bpe/vocab.json",merges_filename="stein_tokenizer_bpe/merges.txt")
# print(f"tokenizer size {tokenizer.get_vocab().__len__()}")#tokenizer.pad_token = tokenizer.eos_token  # Set <pad> token as <eos> for simplicity
# exit()
# Create a Dataset from the tokenized chunks
import numpy
np_arr:numpy.ndarray  = numpy.load("C:/data/nlp/dataset.npy")
np_arr.astype(int)
ds_start_idx= 4092
dataset     = TokenizedDataset(list(np_arr)[ds_start_idx:],n_positions=config.n_positions)
evalset     = TokenizedDataset(list(np_arr)[:ds_start_idx],n_positions=config.n_positions)

# # 6. Collate function to pad sequences to the same length
# def collate_fn(examples):
#     return tokenizer.pad(
#         {'input_ids': [example['input_ids'] for example in examples]},
#         padding=True,
#         return_tensors="pt"
#     )

# 7. Set up training arguments (optimized for speed)
training_args = TrainingArguments(
    output_dir="./scratch_gpt2_youtube_script",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,  # Adjust based on your GPU/CPU capacity
    num_train_epochs=2,  # Increase if needed, 3 is a minimal viable setting
    save_steps=128,  # Save model checkpoints every 500 steps
    save_total_limit=2,  # Keep only 2 checkpoints
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",  # Evaluate during training
    eval_steps=128,
    do_train=True,
    do_eval=True,
    bf16=torch.cuda.is_available(),  # Use 16-bit precision if on GPU
    optim="sgd",  # Use AdamW optimizer
    warmup_steps=3,  # Warm up learning rate for the first few steps
    gradient_accumulation_steps=4,
    include_tokens_per_second=False
    
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=evalset
    #data_collator=collate_fn
)

# 9. Train the model
trainer.train()

# Save the final model
model.save_pretrained("./scratch_gpt2_youtube_script_final")
tokenizer.save_pretrained("./scratch_gpt2_youtube_script_final")
