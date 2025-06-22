from tok import filter_by_topic, tokenize_corpus,load_tokenizer,train_tokenizer
from crawl import clean_fineweb
from language_utils import normalize_text
from training import *
import numpy 
import tokenizers 
import json
import html 
import re 
import pandas 


class StackItem:

    def __init__(self,q="",a=[],p_id=None):
        self.question       = q
        if not a:
            self.answers    = [] 
        else:
            self.answers    = {a['id'],a['text']}
        
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

def remove_pre_blocks(text):
    return re.sub(r'<pre\b[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)


def clean_html(text:str):
    print("Original:\n" + text + "\n\n")
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
    text        = text.replace('<code>','<|code|>').replace('</code>','<|endcode|>')

    input("Fixed:\n"+text + "\n\n\n\n\n")


def repair_tokens():

    for file in [os.path.join(ULTRA,fname) for fname in os.listdir(ULTRA)]:
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
    root        = 'C:/data/nlp/stack exchange'
    
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
    

    for key in stackdata:
        stackdata[key] = stackdata[key].to_string()

    with open("C:/data/nlp/stackdata.json",'w',encoding='utf_8') as writefile:
        writefile.write(json.dumps(stackdata))


def repair_academic():
    root        = "C:/data/nlp/academic"

    good_ac     = []
    writepath   = "C:/data/nlp/academic.txt"

    for file in [os.path.join(root,fname) for fname in os.listdir(root)]:
        
        with open(file,'r',encoding='utf_8') as readfile:
            contents    = readfile.read()

            good_ac.append(contents)
                
    with open(writepath,'w',encoding='utf_8') as writefile:
        writefile.write(f'{END_TOKEN}'.join(good_ac) + END_TOKEN)


def repair_books():
    root        = "C:/data/nlp/gutenberg/books"

    good_ac     = []
    writepath   = "C:/data/nlp/books.txt"

    for file in [os.path.join(root,fname) for fname in os.listdir(root)]:
        if file.split('\\')[-1][0] in '0123456789':
            continue 

        with open(file,'r',encoding='ascii') as readfile:
            contents    = readfile.read()

            good_ac.append(contents)
                
    with open(writepath,'w',encoding='utf_8') as writefile:
        writefile.write(f'{END_TOKEN}'.join(good_ac) + END_TOKEN)


def repair_yt():
    og_path     = "C:/data/nlp/yt_ascii"

    good_yt     = []
    writepath   = "C:/data/nlp/youtube.txt"

    for file in [os.path.join(og_path,fname) for fname in os.listdir(og_path)]:
        
        with open(file,'r',encoding='ascii') as readfile:
            contents    = readfile.read()
            data        = json.loads(contents)
            transcript  = data['transcript']
            quality     = data['transcribed']

            if quality:
                good_yt.append(clean_youtube_transcript(transcript))
                
    with open(writepath,'w',encoding='utf_8') as writefile:
        writefile.write(f'{END_TOKEN}'.join(good_yt) + END_TOKEN)


def generate_datasources():
    repair_stack()
    repair_academic()
    repair_books()
    repair_yt()


if __name__ == '__main__':
    #Clean fineweb
    clean_fineweb(min_score=.95)
    filter_by_topic()
    #Perform whitelisting 
    #Get all curated data sources
    generate_datasources()


    # #Tokenize based on new whitelist 
    FINAL_SIZE              = 32768
    RESERVED_TOK            = 4
    TOK_NAME                = f'tokenzier2'
    #train_tokenizer(FINAL_SIZE-RESERVED_TOK,TOK_NAME,db=INTER)

    # #Create final data
    #tokenize_corpus(TOK_NAME,db=INTER,tok_db=ULTRA,n_workers=4)

    repair_tokens()

