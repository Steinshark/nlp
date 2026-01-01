from training import *

OPEN_BRACKET    = '['
CLOSE_BRACKET   = ']'
corrections     = {'OPEN_BRACKET':'[',"CLOSE_BRACKET":']'}



def expand_prompts(prompt_completion_pair:list[tuple[str,str]]):
    completions     = []

    if isinstance(prompt_completion_pair[0],str):
        return [(prompt_completion_pair[0],prompt_completion_pair[1])]
    
    for prompt in prompt_completion_pair[0]:
        for completion in prompt_completion_pair[1]:
            completions.append((prompt,completion))
    
    return completions


def train_on_SQuAD():
    model = None


if __name__ == '__main__':

    import pprint 
    writer = open("C:/gitrepos/cloudGPT/finetune/training2.txt",'w',encoding='utf_8')

    all_resp    = []
    finetune_data = eval(open("C:/gitrepos/cloudGPT/finetune/training1.txt",'r',encoding='utf_8').read())
    for resp in finetune_data:
        all_resp += expand_prompts(resp)
    
    writer.write(json.dumps(all_resp))
    #writer.write('[\n')

    # for a,b in all_resp:
    #     writer.write(f'({a},{b}),\n')
    # print(all_resp[0])

    #print(f"generated {len(all_resp)} pairs -> {len(''.join([a+b for a,b in all_resp]))}")
    writer.close()
    
    
    