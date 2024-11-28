import torch.optim
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from tokenizers.implementations import ByteLevelBPETokenizer
from dataset import  TokenizedDataset, InfSampler, create_token_file, TextFileDataset
import torch    
from torch.utils.data import DataLoader
import time 
import os 
import random 
import numpy
import argparse
from torch.nn import CrossEntropyLoss
import torch
from matplotlib import pyplot as plt 
import json 

_KEYTOKENS      = []
def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    #return loss_per_sample.mean()
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).type(torch.float) for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss


DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T_TYPE      = torch.float


class GPTSteinshark(GPT2LMHeadModel):
    

    def __init__(self,
                 input_size=64,
                 vocab_size=1024,
                 n_embed=64,
                 n_layer=16,
                 n_head=4,
                 act_fn="gelu_new",
                 name="steinshark1",
                 tokenizer=ByteLevelBPETokenizer
                 ):
        
        #Create config for the model 
        self.config             = GPT2Config(vocab_size=vocab_size,
                                             n_positions=input_size,
                                             n_embd=n_embed,
                                             n_layer=n_layer,
                                             n_head=n_head,
                                             activation_function=act_fn,
                                             resid_pdrop=.05,
                                             attn_pdrop=.05,
                                             embd_pdrop=.05,
                                             torch_dtype=T_TYPE,
                                             layer_norm_epsilon=1e-5,
                                             scale_attn_by_inverse_layer_idx=True
                                             )

        #Create the model
        super(GPTSteinshark,self).__init__(self.config)
        self.train_device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.train_device)
        self.tokenizer          = tokenizer
        self.n_positions        = input_size
        self.vocab_size         = vocab_size
        self.name               = name
        self.warmup             = True
        self.train_iters        = 0
        self.n_params           = sum([p.numel() for p in self.parameters()])

        self.metadata           = {"tok_trained_on":0,"epochs_complete":0,"iters_trained_on":0}


    def train_stein(self,   
              n_iter=2**17,
              bs=4,
              warmup_bs=16,
              lr=.0002,
              warmup_lr=.0001,
              wd=.01,
              warmup_steps=2048,
              sample_text='my cat is',
              accu_steps=16,
             temp=.85,
              ds_root=""
              ):

        #Data Vars
        self.train_bs                   = bs
        self.warmup_bs                  = warmup_bs
        self.accu_steps                 = accu_steps
        self.eff_bs                     = bs*accu_steps

        #Train Vars
        self.total_training_iters       = n_iter
        self.lr                         = lr
        self.warmup_lr                  = warmup_lr
        self.nominal_lr                 = lr
        self.wd                         = wd 
        self.warmup_steps               = warmup_steps
        self.warming_up                 = True
        self.continue_training          = True 

        #Telemetry vars
import torch.optim
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from tokenizers.implementations import ByteLevelBPETokenizer
from dataset import  TokenizedDataset, InfSampler, create_token_file, TextFileDataset
import torch    
from torch.utils.data import DataLoader
import time 
import os 
import random 
import numpy
import argparse
from torch.nn import CrossEntropyLoss
import torch
rom matplotlib import pyplot as plt 
import json 

_KEYTOKENS      = []
def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    #return loss_per_sample.mean()
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).type(torch.float) for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weimport requests
import random
import time 
from matplotlib import pyplot as plt 
import re 
import json
import os 
import zlib
import unidecode
#import wikipedia


#<td>Page views in the past 30 days</td><td><div class="mw-pvi-month"><a rel="nofollow" class="external text" href="
# title_reg   = '([A-z]|[0-9]|"|:|\(|\)|\.| )'
 OREGEX_KEY   = f'((<a href="\/wiki\/)({title_reg}*)((" title=){title_reg}*)(">)({title_reg}*)(<\/a>))'

# REGEX_KEY   = f'((<a href="\/wiki\/)(([A-z]|[0-9]|:|\(|\))*)((" class=)(([A-z]|-|[0-9]|"|:|\(|\))*))*((" title=)([A-z]|[0-9]|"|:|\(|\)| )*)(">)(([A-z]|[0-9]|"|:|\(|\)| )*)(<\/a>))'
# CITE_KEY    = f'((<sup id=")(([A-z]|[0-9]|_|-|\.)*)(" class="reference"><a href="#cite_note-)(([A-z]|[0-9]|_|-|\.)*)(">)(([A-z]|[0-9]|#|;|&|\.)*)(<\/a><\/sup>))'
# CODE_KEY    = f'(<([^>])*>)'


HEADERS     = {
"Referer":"https://en.wikipedia.org/wiki/Timothy_Williamson",
"Sec-Ch-Ua":'"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
"Sec-Ch-Ua-Mobile":'?0',
"Sec-Ch-Ua-Platform":"Windows",
"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def replacer(match:re.Match):
    text        = match.group()
    text        = text.split('title="')[1].split('"')[0]
    return text

def wiki_html_parse(pagetext:str):
    #parse all paragraphs
    paragraphs      = pagetext.split('<p>')[1:]
    paragraphs      = [p.split('</p>')[0].strip().replace('&#91;','[').replace('&#93;',']').replace("\n"," ") for p in paragraphs]  

    paragraphs      = [re.sub(REGEX_KEY,replacer,p) for p in paragraphs]

    for rule in [CITE_KEY,CODE_KEY]:
        paragraphs      = [re.sub(rule,'',p) for p in paragraphs]
    
    parsed_text     = '\n'.join(paragraphs)

    #Remove random large spaces
    while "  " in parsed_text:
        parsed_text = parsed_text.replace("  "," ")

    return parsed_text
    #input(f"paragraphs:\n{paragraphs}")

def url_encode(url:str):
    url     = url.replace('https://en.wikipedia.org/wiki/','').replace(":",'_').replace(";","_")
    return url

def get_pageviews(pagetext:str):

    #Check page stats
    pretext_status  = pagetext.split('" title="More information about this page"><span>Page information')[0]
    pretext_status  = pretext_status.split('<a href="')[-1]

    #At this point should be the url 
    info_url        = "https://en.wikipedia.org" + pretext_status.replace("amp;","")



    try:
        response    = requests.get(url=info_url,timeout=2,headers=HEADERS)
        response.raise_for_status()
        text        = response.text 

        #Grab stat
        text        = text.split('<td>Page views in the past 30 days</td><td><div class="mw-pvi-month"><a rel="nofollow" class="external text" href=')[1]
        text1        = text.split('">')[1]
        text2        = text1.split('</a><')[0].rstrip().replace(" ","").replace(',','')
        views       = int(text2)
        return views 
    except ValueError as ve:
        print(f"got strange on {url}({text1})")
        print(f"what: {ve}")

def scan_wikipedia(url:str,data,depth,view_thresh:int=0,max_n:int=64,timeout:int=2,directory={},storing:str='C:/users/default/temp/wikistash/data'):

    #Check for return conditions
    if depth == 0:
        return 
    if len(data) > max_n:
        return
    if url in data:
        return
    
    #Load by URL
    if url in directory:
        print(f"discovered {url}")
        #Grab info
        url_data    = directory[url]

        #Check for add
        if url_data['n_visits'] > view_thresh:
            data[url]   = {'text':f"LOOKUP{url_data['filepath']}",'n_visits':url_data['n_visits']}

        #Continue with urls 
        urls        = url_data['urls']
    
    #Load manually
    else:

        #Get from request
        response        = None 
        try:
            response    = requests.get(url=url,timeout=timeout,headers=HEADERS)
            response.raise_for_status()
        except ValueError:
            print(f"value error")

        #Grab views
        pagetext        = response.text
        try:
            n_visits        = get_pageviews(pagetext)
        except IndexError:
            return

        #Add to data if over thresh
        if n_visits > view_thresh:
            print(f"added {url}")
            data[url]   = {'text':wiki_html_parse(pagetext),'n_visits':n_visits}

        #Save to directory
        #compressed  = zlib.compress(pagetext.encode(),level=7)
        compressed = wiki_html_parse(pagetext)
        filepath    = os.path.join(storing,url_encode(url))
        with open(filepath,'w',encoding='utf_8') as file:
            file.write(compressed)

        #Grab all urls from page 
        urls            = re.finditer(REGEX_KEY,pagetext)
        urls            = ["https://en.wikipedia.org/wiki/"+url.group().split('a href="/wiki/')[1].split('" title')[0] for url in urls if not 'Category' in url.group() and not 'Help' in url.group() and not 'Special' in url.group() and not 'Wikipedia' in url.group() and not 'Portal' in url.group() and not 'Main_Page' in url.group() and not 'redirect' in url.group()]
        try:
            urls.remove(url)
        except ValueError as ve:
            pass 

        #Add to directory
        directory[url]  = {'filepath':filepath,'n_visits':n_visits,'urls':urls}


    #Continue down tree 
    random.shuffle(urls)
    for url in urls:
        scan_wikipedia(url,data,depth-1,view_thresh=view_thresh,max_n=max_n,timeout=2,directory=directory,storing=storing)

    return


def scan_wikipedia2(url:str,view_thresh:int=0,max_n:int=64,timeout:int=2,directory={},storing:str='C:/users/default/temp/wikistash/data'):

    urls        = [url] 
    data        = {}
    timeouts    = 0
    while urls and len(data) < max_n:

        #chose random url
        url     = urls.pop(random.randint(0,len(urls)-1))

        #Load by URL
        if url in directory:
            #print(f"discovered {url}")
            #Grab info
            url_data    = directory[url]

            #Check for add
            if url_data['n_visits'] > view_thresh:
                data[url]   = {'text':f"LOOKUP{url_data['filepath']}",'n_visits':url_data['n_visits']}

            #Continue with urls 
            urls        = ["https://en.wikipedia.org/wiki/"+u for u in url_data['urls']]
            
        #Load manually
        else:

            #Get from request
            response        = None 
            try:
                response    = requests.get(url=url,timeout=timeout,headers=HEADERS)
                response.raise_for_status()
                pagetext        = response.text
                #Grab views
                try:
                    n_visits        = get_pageviews(pagetext)
                except IndexError:
                    continue

                #Add to data if over thresh
                if n_visits > view_thresh:
                    #print(f"added {url}")
                    data[url]   = {'text':wiki_html_parse(pagetext),'n_visits':n_visits}

                #Save to directory
                compressed  = wiki_html_parse(pagetext)
                filepath    = os.path.join(storing,url_encode(url))
                with open(filepath,'w',encoding='utf_8') as file:
                    file.write(compressed)

                #Grab all urls from page 
                urllist            = re.finditer(REGEX_KEY,pagetext)
                urllist            = ["https://en.wikipedia.org/wiki/"+url.group().split('a href="/wiki/')[1].split('" title')[0] for url in urllist if not 'Category' in url.group() and not 'Help' in url.group() and not 'Special' in url.group() and not 'Wikipedia' in url.group() and not 'Portal' in url.group() and not 'Main_Page' in url.group() and not 'redirect' in url.group()]
                try:
                    urllist.remove(url)
                except ValueError as ve:
                    pass 

                #Add to end of urls 
                urls                += urllist

                #Add to directory
                directory[url]  = {'filepath':filepath,'n_visits':n_visits,'urls':[u.replace('https://en.wikipedia.org/wiki/','') for u in urllist]}
            except ValueError as ve:
                print(f"value error: {ve}")
            except AttributeError:
                print(f"bad attribute on {url }")
            except requests.exceptions.HTTPError:
                print(f"bad url")
            except requests.exceptions.ReadTimeout:
                print(f"bad timeout")
                timeouts += 1
                with open(f"C:/users/default/temp/wikistash/directory.txt",'w',encoding='utf_8') as file:
                    file.write(json.dumps(directory))
                
                if timeouts     == 10:
                    return data
            except requests.exceptions.ConnectTimeout:
                print(f"bad timeout")
                timeouts += 1
                with open(f"C:/users/default/temp/wikistash/directory.txt",'w',encoding='utf_8') as file:
                    file.write(json.dumps(directory))
                
                if timeouts     == 10:
                    return data
            except requests.exceptions.ConnectionError:
                print(f"bad timeout")
                timeouts += 1
                with open(f"C:/users/default/temp/wikistash/directory.txt",'w',encoding='utf_8') as file:
                    file.write(json.dumps(directory))
                
                if timeouts     == 10:
                    return data


            

    return data


def run_grab(start_page:str,max_n:int=32,view_thresh:int=50_000):
    #Create a temporary repository of wiki data
    if not os.path.exists("C:/users/default/temp/wikistash"):
        os.mkdir("C:/users/default/temp/wikistash")
    
    #Load the history dict 
    if os.path.exists("C:/users/default/temp/wikistash/directory.txt"):
        directory   = json.loads(open("C:/users/default/temp/wikistash/directory.txt",'r',encoding='utf_8').read()) 
    else:
        directory   = {}

    visit_data  = [] 
    url         = f"https://en.wikipedia.org/wiki/{start_page}"

    start_dict  = {}
    #scan_wikipedia2(url,start_dict,4,view_thresh=15_000,max_n=8,directory=directory)
    start_dict  = scan_wikipedia2(url,view_thresh=view_thresh,max_n=max_n,directory=directory)
    data        = [v['n_visits'] for v in start_dict.values()]
    links       = list(start_dict.keys())

    
    # #Save links to file 
    # with open(f"C:/gitrepos/nlp/wikilinks.txt",'w',encoding='utf_8') as file:
    #     file.write(json.dumps(links))

    #Save directory
    with open(f"C:/users/default/temp/wikistash/directory.txt",'w',encoding='utf_8') as file:
        file.write(json.dumps(directory))
    

def clean_csv(csv_file="C:/gitrepos/nlp/data/YT-titles-transcripts-clean.csv"):
    from dataset import clean_text

    #video_transcripts   =    [] 

    with open(csv_file,'r',encoding='utf_8') as read_file:


        file_contents   = read_file.readlines()
        for line in file_contents[1:]:
            video_id    = line.split(",")[0]
            #input(f"id={video_id}")
            transcript  = "".join(line.split(",")[2:-7])[1:-1]
            transcript  = clean_text(transcript)
            #video_transcripts.append(transcript)
            
            with open("C:/gitrepos/nlp/yt_ascii/" + video_id+'.txt','w',encoding='ascii') as writefile:
                try:
                    writefile.write(transcript)
                except UnicodeEncodeError:
                    pass


def fetch_urls(html_text:str):
    yt_urls = [it[0] for it in re.findall(r"(watch\?v=([a-z]|[A-Z]|[0-9]|-|_)+)",html_text)]
    return yt_urls


def scrape_yt(start_url="https://www.youtube.com/watch?v=MtGUr21H7UE",subject_start="chess1",lim=20_000):

    all_urls        = set()
    #Get list of all videos found 
    for fname in os.listdir("C:/data/nlp/urls"):
        fname   = os.path.join("C:/data/nlp/urls",fname)
        with open(fname,'r',encoding="utf_8") as readfile:
            for item in json.loads(readfile.read()):
                all_urls.add(item)
    print(f"found {len(all_urls)} existing urls")
    
    url_jackpot     = set()
    save_lim        = 1_000
    queued_urls     = {start_url}


    while True:

        #Get next url
        next_url    = queued_urls.pop()

        try:
            time.sleep(.02+random.random()*2)
            req         = requests.get(url=next_url,timeout=3)

            if random.random() < .1:
                print(f"fetched {len(url_jackpot)} urls")

            if req.status_code == 200:
                html    = req.text 
                discoveries = fetch_urls(html)
                for url in discoveries:
                    
                    if not url in all_urls:
                        queued_urls.add("https://www.youtube.com/"+url)
                        url_jackpot.add("https://www.youtube.com/"+url)

        except requests.Timeout:
            print(f"failed to fetch url {next_url}")
            queued_urls.add(next_url)
            time.sleep(random.randint(2,7))

        if len(url_jackpot) > save_lim or len(url_jackpot) > lim:
            print(f"saved {len(url_jackpot)} urls")
            with open(f"C:/data/nlp/urls/scraped_urls_{subject_start}.json",'w',encoding='utf_8') as writefile:
                writefile.write(json.dumps(list(url_jackpot)))
            save_lim += 1_000

            if len(url_jackpot) > lim:
                print(f"exiting")
                break 


def grab_blogs(csv_file="C:/data/nlp/blogs/blogtext.csv"):
    texts   = [] 

    with open(csv_file,'r',encoding='utf_8') as readfile:

        #Get category line 
        #readfile.readline()

        #Read file
        while readfile:
            blog_text   = readfile.readline()
            input(blog_text)
    #


def transcript_grabber():
    all_urls        = set()

    #Get a set of all videos found 
    for fname in os.listdir("C:/data/nlp/urls"):
        fname   = os.path.join("C:/data/nlp/urls",fname)
        with open(fname,'r',encoding="utf_8") as readfile:
            for item in json.loads(readfile.read()):
                all_urls.add(item)
    print(f"found {len(all_urls)} existing urls")

    #Go through set and download 

    while all_urls:
        next_url    = all_urls.pop()

        next_id     = next_url.replace("https://www.youtube.com/watch?v=","")


def get_gutenberg():

    for i in range(20000):
        i += 1000
        time.sleep(random.random())
        url     = f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt"
        check   = 'Language: English'

        req     = requests.get(url)

        if req.status_code == 200:
            if check in req.text[:10000]:
                with open(f"C:/data/nlp/gutenberg/books/{i}.txt",'w',encoding='ascii') as writefile:
                    writefile.write(unidecode.unidecode(req.text))
            else:
                pass
        else:
            print("bad req")

    
if __name__ == "__main__":

    max_n       = 4
    starters    = ["Mathematics","Computer_Science","United_States","Biology","History","English_language","Square","Ronald_Reagan","Internet","Dog","Economics","Movie","World_War_II","Aircraft_Carrier","Navy","China"]

    searches    = [("https://www.youtube.com/watch?v=tmbZVmXyOXM",'engineering1'),
    ("https://www.youtube.com/watch?v=1iQV_vcFSE0",'software2'),
    ("https://www.youtube.com/watch?v=xjH2B_sE_RQ","ai1"),
    ("https://www.youtube.com/watch?v=sH1PVVJuBtE","engineering2"),
    ("https://www.youtube.com/watch?v=LSvhVPqKXk0","chess3"),
    ("https://www.youtube.com/watch?v=hBMoPUAeLnY","jre2")]

    for search in searches:
        url,subj    = search
        scrape_yt(start_url=url,subject_start=subj,lim=30_000)

