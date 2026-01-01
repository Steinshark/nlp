from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login(token=open('key.secret','r').read())
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")