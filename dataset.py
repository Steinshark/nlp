import multiprocessing.pool
import torch 
from torch.utils.data import Dataset, Sampler
from tokenizer import SteinTokenizer
import numpy
import os 
import random 
import hashlib
import unidecode
import multiprocessing
import sys 
#sys.path.append("youtubeDB")
from youtubeDB.utils import filter_bad_content

class InfSampler(Sampler):

    def __init__(self):

        pass 

    def __iter__(self):
        while True:
            yield 1

    def __len__(self):
        return float("inf") 


class TextFileDataset(Dataset):

    #Assumption is that all files in ds_root are pre_cleaned and ascii encodable
    def __init__(self,ds_root:str,n_positions:int,max_files:int=1_000_000,tokenize_with=None):

        #Get list of filenames
        self.filenames      = [os.path.join(ds_root,file) for file in os.listdir(ds_root)][:max_files]  

        #Load all content
        self.texts          = [open(fname,'r',encoding='ascii').read() for fname in self.filenames]

        self.tokens         = [] 
        #Set class variables
        self.n_positions    = n_positions

        #Start in warmup mode
        self.warmup         = True
        self.train_i        = .05    

        self.tokenize_with  = tokenize_with
        #Perform cleaning operations
        

    def print_stats(self):
       print(f"Created dataset:\n\t\t{len(self.text.split(' '))/1_000_000:.2f}M words\n\t\t{len(self.text)/1_000_000:.2f}M characters\n\t\t{len(self.tokenized_text)/1_000_000:.2f}M tokens\n\t\t{len(set(self.text))} unique chars\n\n")


    def as_dict(self):
        return {"text":self.text[i] for i in range(len(self.text))}
        
    def tokenize(self,tokenizer):
        print(f"attempting tokenization of {len(self.texts)}")
        token_chunks    = [tokenizer.encode(text).ids for text in self.texts]
        print(f"created results, appending")
        for chunk in token_chunks:
            self.tokens += chunk
        self.tokens     = numpy.asarray(self.tokens)

        print(f"self.tokens is {self.tokens.shape}")


            

    #returns text_ids and attention_mask
    def __getitem__(self,i):
        
        #Pick random window 
        start_i                 = random.randint(0,int(len(self.text)*self.train_i)-self.n_positions)

        return self.tokenized_text[start_i:start_i+self.n_positions]
        window                  = random.randint(1,self.n_positions)
        window                  = self.n_positions*10
        return self.text[start_i:start_i+window]

        
    def __len__(self):
        return self.size


    def clean_text(self):

        #All lower 
        #self.text           = self.text.lower()
        #Remove double newlines
        changes     = {
            '\xa3':'L',
            '\xa0':' ',
            '\u2019':"'",
            '\xb0':'degrees',
            '\u266a':'[music]',
            '\u2026':'...',
            '\U0001f60a':':)',
            "\x2018":"'",
            "\u2018":"'",
            "\xe9":"e",
            "\u20ac":'Euro',
            "\u2044":"/",
            "\xe8":'e',
            "\xf1":"n",
            "\u201c":'"',
            "\u201d":'"',
            "\xed":'i',
            "\xc5":'A',
            "\xf3":'o',
            "\u2014":'-',
         #   '\n':' ',
            "'":'',
            'é':'e',
            "♪":'[music]',
            '\xa0':' ',
            " uh ": " ",
            " i i ": " ",
            " the the ": " ",
            " um ": " ",
            " it's it's ": " it's"
        }

        for x,y in changes.items():
            self.text   = self.text.replace(x,y).lower()

    @staticmethod
    def cleantext(text):

        text = text.lower()
        changes     = {
            '\xa3':'L',
            '\xa0':' ',
            '\u2019':"'",
            '\xb0':'degrees',
            '\u266a':'[music]',
            '\u2026':'...',
            '\U0001f60a':':)',
            "\x2018":"'",
            "\u2018":"'",
            "\xe9":"e",
            "\u20ac":'Euro',
            "\u2044":"/",
            "\xe8":'e',
            "\xf1":"n",
            "\u201c":'"',
            "\u201d":'"',
            "\xed":'i',
            "\xc5":'A',
            "\xf3":'o',
            "\u2014":'-',
         #   '\n':' ',
            "'":'',
            'é':'e',
            "♪":'[music]',
            '\xa0':' ',
            " uh ": " ",
            " i i ": " ",
            " the the ": " ",
            " um ": " ",
            " it's it's ": " it's"
        }
        for x,y in changes.items():
            text   = text.replace(x,y)
        
        return text
    
    def create_tokenized(self):
        pass


    def get_iter(self,max_i=100_000_000):
        i   = 0 
        for item in self.texts.split(self.eos_token):
            yield item
        return
        text_files          = [open(fname,'r',encoding='utf_8').read().lower() for fname in self.filenames]
        print(f"number of text files: {len(text_files)}")
        for text in text_files:
            i += 1 
            if i > max_i:
                break
            yield text


    def save_to_file(self,fname:str):
        with open(fname,'w',encoding='utf_8') as file:
            file.write(self.texts)
        return 


class TokenizedDataset(Dataset):

    def __init__(self,tokens,n_positions):
        self.tokens         = tokens 
        self.n_positions    = n_positions 

        self.n_tokens       = len(self.tokens)
        self.warmup         = True
    

    def __getitem__(self, index)->dict[str,torch.Tensor]:
        #Pick random start 

        end_point           = len(self.tokens) - self.n_positions if not self.warmup else int(len(self.tokens)*.02)
        start_i             = random.randint(0,end_point)

        token_seq           = numpy.asarray(self.tokens[start_i:start_i+self.n_positions])
        token_seq_torch     = torch.from_numpy(token_seq).type(torch.long)

        return {"input_ids":token_seq_torch,"labels":token_seq_torch}
    
    def __len__(self):
        return len(self.tokens) // self.n_positions

 
#All files are assumed to be cleaned
def create_token_file(ds_root,input_size,tokenizer):

    #Gather all filenames
    filenames       = [os.path.join(ds_root,file) for file in os.listdir(ds_root)]  

    #Create list of all texts
    texts           = [open(fname,'r',encoding='utf_8').read()+ "<|endoftext|>" for fname in filenames]
    tokens          = [] 

    print(f"tokenizing texts")
    for text in texts:
        tokens += tokenizer.encode(text).ids

    print(f"saving np")
    np_arr:numpy.ndarray  = numpy.asarray(tokens)   
    np_arr.astype(int)
    numpy.save("dsnumpy.npy",np_arr)
    print(f"created token set {np_arr.shape}")


def get_yt_captions(ytdump_file:str='ytdump.html'):
    import youtube_transcript_api
    from youtube_transcript_api import YouTubeTranscriptApi
    import time 


    #Parse file for links 
    with open(ytdump_file,'r',encoding='utf_8') as read_file:

        filetext        = read_file.read()
        splittext       = filetext.split("/watch?v=")
        yt_ids          = set([text.split('"')[0] for text in splittext][1:])
        cleaned_ids     = []
        for ytid in yt_ids:
            if '&' in ytid:
                ytid = ytid.split("&")[0]
            if "t=" in ytid:
                ytid = ytid.split("t=")[0]

            cleaned_ids.append(ytid)


        grabber         = YouTubeTranscriptApi()
        for id in cleaned_ids:
            if os.path.exists(f"yt_captions/{id}.txt"):
                print(f"already had {id}")
                continue
            try:
                video_text   = " ".join([l['text'] for l in grabber.get_transcript(id)])
                print(f"found {len(video_text.split(' '))} words")
                with open(f"yt_captions/{id}.txt",'w',encoding='utf_8') as write_file:
                    write_file.write(video_text)
                    write_file.close()
            except youtube_transcript_api._errors.NoTranscriptFound:
                pass
            except youtube_transcript_api._errors.TranscriptsDisabled:
                pass


#Provided a list of directories of text files,
# combine these into train_dir based on contents 
def prep_data_for_training(desired_sources:list[str],final_dir="C:/data/nlp/train_dir",eot_token="<|endoftext|>"):
    
    #Clear dataroot 
    for file in [os.path.join(final_dir,fname) for fname in os.listdir(final_dir)]:
        os.remove(file)
    

    hasher                  = hashlib.md5()
    
    #stats 
    chars                   = 0 
    words                   = 0 
    for rootdir in desired_sources:

        for fname in os.listdir(rootdir):
            
            #Get absolute path
            fpath_load      = os.path.join(rootdir,fname)
            #Load contents
            with open(fpath_load,'r',encoding='utf_8') as readfile:
                contents    = readfile.read()
                contents    = clean_individual_text(contents)
            
            #Ensure no duplicate contents
            hasher.update(contents.encode())
            content_hash    = hasher.hexdigest()
            fpath_save      = os.path.join(final_dir,content_hash+".txt")

            #Ensure not saving empty file
            if not filter_bad_content(contents):
                continue

            if not os.path.exists(fpath_save):
                chars       += len(contents)
                words       += len(contents.split(" "))

                #Write contents 
                with open(fpath_save,'w',encoding='ascii') as writefile:
                    writefile.write(contents+eot_token)
    texts                   = len(os.listdir(final_dir))

    print(f"gathered texts\n\tchars: {chars//1_000_000}M\n\twords: {words//1_000_000}M\n\ttexts: {texts}")


#Create a ASCII version of the transcripts 
# also take out stops words and other stupid stuff
def clean_individual_text(contents):

    #Remove all double newlines 
    contents    = unidecode.unidecode(contents)
    contents    = contents.replace("\n\n","\n")

    #Preserve python indents 
    contents    = contents.replace("    ",'|PYTHONINDENT|')
    while "  " in contents:
        contents = contents.replace("  "," ")
    contents    = contents.replace('|PYTHONINDENT|',"    ").lower()
    
    return contents



def resave():

    changes     = {
        '\xa3':'L',
        '\xa0':' ',
        '\u2019':"'",
        '\xb0':'degrees',
        '\u266a':'[music]',
        '\u2026':'...',
        '\U0001f60a':':)',
        "\x2018":"'",
        "\u2018":"'",
        "\xe9":"e",
        "\u20ac":'Euro',
        "\u2044":"/",
        "\xe8":'e',
        "\xf1":"n",
        "\u201c":'"',
        "\u201d":'"',
        "\xed":'i',
        "\xc5":'A',
        "\xf3":'o',
        "\u2014":'-',
        " um ": " ",
        " uh ": " ",
        " i i ": " i ",
        'ç':"c",
        'с':"c",
        'π':"pi",
        '≅':"~=", 
        '™':"TM",
        '𝑛':"n",  
        '𝑃':"P", 
        'ℙ':"P", 
        '−':"-", 
        '²':"^2", 
        '𝜋':"pi", 
        '𝐸':"E", 
        '~':"~", 
        'γ':"gamma", 
        '′':"`", 
        '¹':"^1", 
        '⁵':"^5", 
        'в':"B", 
        '𝐺':"G", 
        '₂':"_2", 
        '∀':" for all ", 
        'м':"M",
        '∃':" there exists ",
        'Δ':"Delta", 
        '𝜃':"Theta", 
        '‽':"?!", 
        '𝛿':"sigma", 
        'ő':"o", 
        '𝐻':"H", 
        '�':"?",
        '∈':" element of ",
        '₁':"_1",
        'δ':"sigma", 
        '∎':"[]", 
        '⊗':"X", 
        'ɸ':"phi", 
        'ν':"v", 
        'ℕ':"N", 
        '\u2009':"?",
        '𝐷':"D", 
        '·':" dot ",
        'ä':"a",
        '̶':"?", 
        '⁰':"degrees", 
        'É':"E",
        'à':"a",
        'е':"e",
        'д':"D",
        '×':"x",
        '→':"->", 
        'ö':"o",
        'Ο':"O",
        '𝐶':"C",
        '𝑎':"alpha",
        'ú':"u",
        'т':"T",
        '𝐹':"F",
        '½':"1/2",
        'ℝ':"R", 
        'θ':"Theta", 
        'έ':"e", 
        'ô':"o", 
        '³':"^3", 
        'á':'a', 
        '𝓁':"l", 
        '´':"`", 
        'ń':"n", 
        '⅓':"1/3", 
        'ï':"l", 
        '･':"dot", 
        '–':"-", 
        '𝐵':"B", 
        '∩':"intersection", 
        '𝑏':"b", 
        '∞':"infinity", 
        '∂':"b", 
        '¡':"!", 
        'ü':"u", 
        '⁴':"^4", 
        'ᵢ':"_i", 
        '♫':"[music]", 
        'υ':"v", 
        '😲':":)",
        'ë':"e",
        'ã':"",
        'ā':"a",
        'š':"s",
        'ř':"r",
        "ō":"o",
        "õ":"o",
        'й':"n",
        'ì':"i",
        'ī':"i",
        'Š':"S",
        'ù':"u",
        '鼎':"?",'Н':"?",'у':"?",
        '騎': "?",'幡': "?",'工': "?",
        '昌': "?",'玉': "?",'п': "?",
        '進': "?",'高': "?",'崎': "?",
        '所': "?",'С': "?",'橋': "?",
        '梁': "?",'新': "?",'木': "?",
        'я': "?",'酒': "?",'空': "?",
        '\ufeff': "?",'電': "?",'星': "?",
        '和': "?",'豹': "?",'梅': "?",
        '枸': "?",'н':"?",'許': "?",
        '知': "?",'東': "?",'水': "?",
        '吉': "?",'鉧': "?", '米': "?",
        '運': "?",'州': "?",'可': "?",
        '麒': "?",'Ж': "?",'Е': "?",
        '造': "?",'千': "?",'茅': "?",
        '陳': "?",'耀': "?",'胡': "?",'ズ': "?",'番': "?",'柚': "?",
        '德': "?",'成': "?",'潘': "?",'壹': "?",
        '豊': "?",'白': "?",'櫻': "?",'国': "?",'思': "?",
        '𝐴': "?",'孟': "?",'區': "?",'吻': "?",
        '紹': "?",'海': "?",'份': "?",'В' :"?",
        '井': "?",'ー': "?",'ス': "?",'竹': "?",
        '麟': "?",'盛': "?",'定': "?",'門': "?",
        '攤': "?",'河': "?",'К': "?",'的':"?",
        '鮭': "?",'義': "?",'ь': "?",'鐵': "?",
        '́': "?",'ы': "?",'研': "?",'股': "?",'号': "?",'日': "?",'書': "?",'台': "?",
        '立':"?",'鑑': "?",'學': "?",'限': "?",
        'ш': "?",'房': "?",'鉤': "?",'л': "?",
        '傅': "?",'  春': "?",'貴': "?",'А': "?",
        '製': "?",'文': "?",'彦': "?",'政': "?",'花': "?",
        'И': "?",'デ': "?",'小': "?",'リ': "?",'淵': "?",'美': "?",
        '￼': "?",'瀬': "?",'藤': "?",'盧':"?",'蔭':"?",'燒': "?",
        '大': "?",'鵬': "?",'公': "?",'辰': "?",'ラ': "?",'華': "?",'陽': "?",'科': "?",
        '翎': "?",'鋼': "?",'帰': "?",'际': "?",'偈':"?",' 八': "?",
        '印':"?",'六': "?",'璿': "?",'ž': "?",'制': "?",'零': "?",
        '间': "?",'廠': "?",'ド': "?",'集': "?",'и': "?",'к': "?",
        '蓝': "?",'交':"?",'發': "?",'有': "?",'川': "?",'р': "?",
        'х': "?",'園': "?",'斗': "?",'鹿': "?",'争': "?",'子': "?",
        'Ф': "?",'ч': "?",'曹': "?",'赤': "?",'團': "?",'澳': "?",
        '西': "?",'本': "?",'レ': "?",'福': "?",'李': "?",'ニ': "?",
        'サ': "?",'衡': "?",'荣': "?",'士': "?",'司': "?",'松': "?",
        'б': "?",'た': "?",'о':"?",'ィ':"?",'見':"?",'ク': "?",
        '通': "?",'ギ': "?",'鯤': "?",'航': "?",'寧': "?",'ら': "?",
        '箭': "?",'國': "?",'ン': "?",'京': "?",'客': "?",'牂': "?",
        '天': "?",'孫': "?",'а': "?",'坪':"?", 'ě':"?", '鉄':"?",
        ' 平':"?",'Т':"?", '欣':"?", '榮':"?", '中':"?", 'г':"?", 
        '酱':"?", '柯':"?", '院':"?",'春':"?", '平':"?", '八':"?",
        '𝑧':"z", '⅔':"2/3", '¼':"1/4", 'ω':"w", '𝑤':"w"
}


    toks        = set()
    good_toks   = set() 
    fail_flag   = False
    for file in os.listdir("yt_captions"):
        fname = f"yt_captions/{file}"
        if os.path.exists(fname.replace("yt_captions","yt_ascii")):
            #os.remove(fname)
            continue

        with open(fname,'r',encoding='utf_8') as readfile, open(fname.replace("yt_captions","yt_ascii"),'w',encoding="ascii") as writefile:
            contents    = readfile.read()

            for x,y in changes.items():
                contents    = contents.replace(x,y)
            try:
                writefile.write(contents.lower())
                [good_toks.add(t) for t in set(contents)]
            except UnicodeEncodeError:
                fail_flag = True
                [toks.add(t) for t in set(contents)]
                pass
    if fail_flag:
        print(toks-good_toks)   
    else:
        print("perfect")


def find_by_topic(final_dir="C:/data/nlp/train_dir",topic_keywords={"project":2,"application":5,"variable":6,"python":25,"c++":25,"cpp":25,"g++":25,"pytorch":25,"coding":25,"program":25,"parser":25,"computer":8,"neural network":25,"computer science":25,"comput":4,"byte":7,"tutorial":6,"code":6,"developer":12}):


    scores  = {} 
    root    = "C:/data/nlp/"
    for fpath in [os.path.join(final_dir,fname) for fname in os.listdir(final_dir)]:
        
        #Get contents score DENSITY 
        with open(fpath,'r') as readfile:
            contents    = readfile.read()
            file_score  = 0 
            indv_score  = [] 
            #Get score 
            for topic,score in topic_keywords.items():
                indv_score.append((contents.count(topic)*score) ** 1/2)
            
            fscore  = 1 
            for iscore in indv_score:
                fscore *= iscore if iscore else 1
            scores[fpath]   = fscore / len(contents)

    sorted_scores   = sorted(scores,key=lambda x: scores[x],reverse=True)

    for fpath in sorted_scores[:10]:
        print(f"{fpath}-> {scores[fpath]}")
    
            
if __name__ == "__main__":
    prep_data_for_training(desired_sources=["C:/data/nlp/yt_ascii"])#,"C:/data/nlp/gutenberg/books","C:/data/nlp/academic","C:/data/nlp/code/randpython_files"])
    exit()
