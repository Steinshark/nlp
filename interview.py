from transformers import AutoModelForCausalLM, AutoTokenizer,MistralModel, StoppingCriteriaList,StoppingCriteria
import torch


#Prep model and tokenizer 
tokenizer                   = AutoTokenizer.from_pretrained("openchat/openchat-3.5-0106")
model:AutoModelForCausalLM  = AutoModelForCausalLM.from_pretrained("openchat/openchat-3.5-0106").eval().half().cuda()
model.do_sample             = True



class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False
stop_words = [";"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])



working_file                = ''
last_command                = ''
last_output                 = ''

task                        = """
create a program used as a math API with the following functions:
name: factorial, args: {num}, description: returns factorial of num 
name: max, args: {a1,a2}, description: returns the max of a1,a2
name: caregiver, args: {name}, description: returns a random int between 0 and length of name"
"""
def gen_prompt():

    #Prepare conversation
    prompt = """*You are an autonomous actor that develops python code. You interact with the IDE via the following phrases: {[write],[erase],[finish]}. 
'[write]' followed by a space, code enclosed in double quotes, a space, and a semicolon at the end will append the text to the working file. ex:
[write] "hello world" ;
'[erase]' followed by an integer, a space,  and a semicolon will remove that number of characters starting from the end of the file. ex:
[erase] 10 ;
'[finish]' indicates the program complies with the task ex:
[finish]
RULES:
only single-quotes for written code is allowed

TASK:""" + task + """
HISTORY:
current file contents are: 
""" + f"\n'{working_file}'\n\n\n" + """

last command was:
""" + f"\n'{last_command}'" + """ 

last output was:
""" + f"\n{last_output}" + """

NEXT INTERACTION:
"""

    return prompt


model.temperature   = .8

stop_tokens         = [tokenizer.encode(";")]

for _ in range(8):
#Interviewer asks question:
    with torch.no_grad():

        encoded         = torch.tensor(tokenizer.encode(gen_prompt()),dtype=torch.long,device='cuda').unsqueeze(dim=0)
        next_command    = tokenizer.decode(model.generate(input_ids=encoded,max_new_tokens=512,stopping_criteria=stopping_criteria)[0])

        #interpret command
        last_prompt     = gen_prompt()

        if "[write]" in next_command:
            next_command    = next_command.replace(last_prompt,"").replace('[write] "',"").replace(" ;","").replace(r"\n","\n").replace(r"\t","\t").replace('"','').replace("<s> ","")
            working_file    += next_command 

        elif "[erase]" in next_command: 
            next_command    = int(next_command.replace(last_prompt,"").replace("[erase] ","").replace(" ;",""))
            working_file    = working_file[:-next_command]

        elif "[finish]" in next_command:
            print(f"ended with:\n{working_file}\n\n\n\n")


            task            = """
create a program used as a math API with the following functions:
name: factorial, args: {num}, description: returns factorial of num 
name: max, args: {a1,a2}, description: returns the max of a1,a2
name: caregiver, args: {name}, description: returns a random int between 0 and 420"
"""
        
        print(f"\n\n\n\n\n{working_file}")
