import random 
from data_cleaning import filter_by_topic, clean_fineweb, clean_fineweb_streamed,filter_wikipedia
from crawl import clean_fineweb
from language_utils import normalize_text, compiled_boundary_corrections,compiled_nonboundary_corrections
from training import *
from tok import tokenize_corpus
import numpy 
import json
import html 
import re 
import pandas 
from tokenizers.implementations import ByteLevelBPETokenizer
import multiprocessing
import datasets 
import warnings 
warnings.filterwarnings("ignore")
import ast
from datagrab import download_huggingface
import os 

class StackItem:

    def __init__(self,q="",a=[],p_id=None):
        self.question       = q
        if not a:
            self.answers    = [] 
        else:
            self.answers    = {a['id']:a['text']}
        
        self.p_id           = p_id 
    
    def add_answer(self,a):
        a_id                = a['id']
        a_text              = a['text']
        self.answers[a_id]  = a_text

    def q_exists(self):
        return bool(self.question) 

    def exists(self):
        return bool(self.answers)
    
    def populate_q(self,q):
        self.question = q 
    
    def to_string(self):
        return {"question":self.question,"answers":self.answers,"id":self.p_id}
    
    @staticmethod
    def from_string(d):
        si      = StackItem(d['question'],[],d['id'])
        for item in d['answers']:
            si.answers.append(item)
        return si

def correct_boundaries(text: str) -> str:
    for pattern, replacement in compiled_boundary_corrections+compiled_nonboundary_corrections:
        text = pattern.sub(replacement, text)
    return text

#Tokenizes based on the tokens found in CRAWL_DB
def train_tokenizer(vocab_size:int,name:str,db:str=CRAWL_DB) ->ByteLevelBPETokenizer:
    print(f"Training {name} tokenizer size={vocab_size}")
    tokenizer               = ByteLevelBPETokenizer()
    tokenizer.train([os.path.join(db,fname) for fname in os.listdir(db)],vocab_size=vocab_size)

    tokenizer.add_tokens(SPECIAL_TOKENS)

    if not os.path.exists(f"{PATH}/{name}"):
        os.mkdir(f"{PATH}/{name}")

    tokenizer.save_model(f"{PATH}/{name}")
    print(f"\tcomplete - saved as {name}")
    

#Loads tokenizer from default location. Adds the endoftext token
def load_tokenizer(tokenizer_name:str)->ByteLevelBPETokenizer:
    tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename=f"{tokenizer_name}/vocab.json",merges_filename=f"{tokenizer_name}/merges.txt")
    tokenizer.add_tokens([END_TOKEN])
    return tokenizer


def remove_pre_blocks(text):
    return re.sub(r'<pre\b[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)


def clean_html(text:str):
    #Remove all html tags that are fine 
    removers   = ['<p>','</p>','<em>','</em>','<br />', '</pre>']

    #Possibles: <strong> <a href=...>

    for repr in removers:
        text        = text.replace(repr,"")

    replacers   = [('&quot;','"'),('&lt;','<'),('&gt;','>')]

    for pair in replacers:
        a,b         = pair 
        text        = text.replace(a,b)


    text        = remove_pre_blocks(text)

    #Make code its own token 
    text        = text.replace('<code>',CODE_START).replace('</code>',CODE_END)

    return text


def count_comment_and_code_lines(source_code):
    lines = source_code.splitlines()
    total_lines = len(lines)
    comment_lines = 0
    string_lines = 0
    in_triple_quote = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            comment_lines += 1
        elif (stripped.startswith('"""') or stripped.startswith("'''")):
            in_triple_quote = not in_triple_quote
            string_lines += 1
        elif in_triple_quote:
            string_lines += 1

    return comment_lines + string_lines, total_lines


def extract_docstrings(source_code):
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

        


    docstrings = []
    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append(docstring)
    return docstrings


def clean_python(root:str):

    CORNERSTONE_IMPORTS = {
    'numpy', 'torch', 'pandas', 'scipy', 'matplotlib', 'tensorflow', 'sklearn',
    'transformers', 'datasets', 'tokenizers', 'xgboost', 'lightgbm',
    'flask', 'http', 'socket', 'requests', 'argparse', 'sys', 'os', 're', 'logging',
    'json', 'csv', 'pickle', 'yaml', 'h5py', 'sqlite3',
    'pathlib', 'typing', 'unittest', 'pytest', 'click', 'warnings',
    'threading', 'multiprocessing', 'concurrent', 'asyncio',
    'seaborn', 'plotly', 'IPython', 'tqdm', 'shutil', 'glob', 'subprocess',
    'time', 'datetime', "collections", "itertools","functools","bisect","heapq","types",
    'enum'
}

    next_file   = [] 
    n_chars     = 0
    write_i     = 0 

    for file in [os.path.join(root,fpath) for fpath in os.listdir(root)]:
        try:
            data    = pandas.read_parquet(file,engine='pyarrow')
        except Exception as e:
            print(f"file '{file}' failed to open")
            continue
        
        
        for i,d in data.iterrows():
            bypass      = False
            date    = d['max_forks_repo_forks_event_min_datetime']
            if date is None:
                continue
            if int(date[:4]) < 2018 or date is None:
                continue

            length  = d['size']
            if length < 100 or length > 25_000:
                continue

            #Get code segments to search
            code:str  = d['content']
            header  = code[:600]

            #Filter files that have comments in non-latin chars 
            comments        =  [line.split('#',1)[1] for line in code.split("\n") if '#' in line]
            docstrings      = extract_docstrings(code)

            comment_chars   = "".join(docstrings) + "".join(comments)
            comment_set     = set(comment_chars)

            comment_ratio   = len(comment_chars) / length 


            #If comment ratio isnt .1-.7, bypass
            if comment_ratio < .1 or comment_ratio > .7:
                bypass      = True 

            else:
                #If bad chars in comments, bypass
                for char in comment_set:
                    if not char in ALLOWABLE_CHAR or bypass:
                        bypass  = True
                        break

            if bypass:
                continue

            #If no good imports, then pybass
            for inp in CORNERSTONE_IMPORTS:
                if inp in header:
                    code        = normalize_text(code)
                    next_file.append(code + "\n" + END_TOKEN)
                    n_chars += length 
                    #input(code)
                    break
            # else:
            #     #Randomly keep .05 of the files
            #     if random.random() < .05:
            #         code        = normalize_text(code)                
            #         next_file.append(code + "\n" + END_TOKEN)
            #         n_chars += length 
                
            
            if n_chars > 100_000_000:
                with open(f"{TRAINING_TEXT2}/python{write_i}.txt",'w',encoding='utf_8') as writefile:
                    writefile.write("".join(next_file))
                
                next_file = []
                write_i += 1 
                n_chars = 0
            

          
def repair_tokens():

    for file in [os.path.join(TRAINING_TOKENS,fname) for fname in os.listdir(TRAINING_TOKENS)]:
        np_arr  = numpy.load(file,allow_pickle=True)

        # for i in range(len(np_arr)):
        #     if not isinstance(np_arr[i], int):
        #         np_arr[i] = np_arr[i].ids[0]
        
        np_arr  = np_arr.astype(numpy.uint16)
        numpy.save(file,np_arr)


def clean_youtube_transcript(text):
    # Unescape HTML
    text = html.unescape(text)

    # Remove all font tags
    text = re.sub(r'<\/?font[^>]*>', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def repair_stack():
    root        = 'alldata/stack exchange'
    
    stackdata:dict[int,StackItem]   = {}
    
    for fpath in [os.path.join(root,fname) for fname in os.listdir(root)]:

        data    = pandas.read_csv(fpath)

        for id,post in data.iterrows(): 
            post_id         = post['Post Id']
            post_parent_id  = post['Parent Id']
            post_type       = post['PostType']
            data            = post['Body']

            data            = clean_html(data)


            if post_type == 'Question':

                #Add it to the stack 
                if post_id in stackdata:
                    stackdata[post_id].populate_q(data)
                
                else:
                    stackdata[post_id] = StackItem(data,[],post_id)
            
            else:
                answer      = {'id':post_id,'text':data}
                if post_id in stackdata:
                    stackdata[post_id].add_answer(answer)
                else:
                    stackdata[post_id] =  StackItem("",answer,post_parent_id)
    
    
    #Save to db 
    filestackdata   = {}
    for key in stackdata:
        filestackdata[key] = stackdata[key].to_string()
    with open("alldata/stackdata.json",'w',encoding='utf_8') as writefile:
        writefile.write(json.dumps(filestackdata))

    #Now, combine the texts and add to traintext
    traintexts      = [] 
    for id in stackdata:
        data:StackItem      = stackdata[id]
        post_text           = f"Question:\n{data.question}\n\n" 
        for id in data.answers:
            post_text += f'Answer:\n{data.answers[id]}\n\n'
        
        traintexts.append(post_text + END_TOKEN + "\n")

        


    with open(f"{TRAINING_TEXT}/stacktraining.txt",'w',encoding='utf_8') as writefile:
        writefile.write(''.join(traintexts))


def repair_academic():
    root        = "alldata/academic"

    good_ac     = []

    for file in [os.path.join(root,fname) for fname in os.listdir(root)]:
        
        with open(file,'r',encoding='utf_8') as readfile:
            contents    = readfile.read()

            good_ac.append(contents)
                
    with open(f"{TRAINING_TEXT}/academic.txt",'w',encoding='utf_8') as writefile:
        writefile.write(f'{END_TOKEN}'.join(good_ac) + END_TOKEN + "\n")


def repair_books():
    root        = "alldata/books"

    good_ac     = []

    for file in [os.path.join(root,fname) for fname in os.listdir(root)]:
        if file.split('\\')[-1][0] in '0123456789':
            continue 

        with open(file,'r',encoding='utf_8') as readfile:
            contents    = readfile.read()

            good_ac.append(contents)
                
    with open(f"{TRAINING_TEXT}/books.txt",'w',encoding='utf_8') as writefile:
        writefile.write(f'{END_TOKEN}'.join(good_ac) + END_TOKEN + "\n")


def repair_yt():
    og_path     = "alldata/yt_clean"

    good_yt     = []

    for file in [os.path.join(og_path,fname) for fname in os.listdir(og_path)]:
        
        with open(file,'r',encoding='utf_8') as readfile:
            contents    = readfile.read()
            good_yt.append(clean_youtube_transcript(contents))
                
    with open(f"{TRAINING_TEXT}/youtube.txt",'w',encoding='utf_8') as writefile:
        writefile.write(f'{END_TOKEN}'.join(good_yt) + END_TOKEN + "\n")


def get_data():

    from datasets import load_dataset
    from huggingface_hub import login 
    login(token=open('key.secret','r').read())

    needed_files                = [f'003_0000{i}.parquet' for i in range(10)]
    load_dataset("HuggingFaceFW/fineweb", data_dir="data/CC-MAIN-2024-51", split="train",data_files=needed_files)


def create_datasources():
    repair_stack()
    repair_academic()
    repair_yt()
    repair_books()


def fix_dataset():
    root    = TRAINING_TEXT
    paths   = [os.path.join(root,fname) for fname in os.listdir(root)]

    with multiprocessing.Pool(4) as pool:
        results = pool.map(normalize_text,paths)

    for result in results:
        print(f"fixed {result}") 


            
            



if __name__ == '__main__':
    #filter_wikipedia()
    #clean_fineweb_streamed()
    #filter_by_topic()
    # clean_python(root='D:/nlp/pythonparquet')
    #clean_python()
    #exit()
    #get_data()
    #download_huggingface('python')
    #exit()
    #Clean fineweb
    #exit()
    #clean_fineweb(min_score=.93)
    #filter_by_topic()
    #create_datasources()
    #clean_python(root='D:/nlp/pythonparquet')
    #Perform whitelisting 
    #Get all curated data sources
    #get_data()
    #exit()

    fix_dataset()
    # #Tokenize based on new whitelist 
    FINAL_SIZE              = 32768
    RESERVED_TOK            = len(SPECIAL_TOKENS)
    TOK_NAME                = f'tokenizer'
    #train_tokenizer(FINAL_SIZE-RESERVED_TOK,TOK_NAME,db=TRAINING_TEXT)

    # #Create final data
    tokenize_corpus(TOK_NAME,db=TRAINING_TEXT,tok_db=TRAINING_TOKENS,n_workers=4)

    #repair_stack()




