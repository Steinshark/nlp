import torch 
from torch.utils.data import Dataset 
from tokenizer import SteinTokenizer

import os 
import random 

class GPTSteinsharkDataSet(Dataset):


    def __init__(self,ds_root:str,n_positions:int,is_dir=True,max_files:int=4096):

        #Get list of filenames and shuffle
        self.filenames      = [os.path.join(ds_root,file) for file in os.listdir(ds_root)][:max_files]  

        #Create the corpus of texts
        self.texts          = [open(fname,'r',encoding='utf_8').read()+ "<|endoftext|>" for fname in self.filenames]
        self.text           = ''.join(self.texts)
        self.clean_text()
        self.tokenized_text = [12,2]
        
        # for file in self.texts:
        #     self.text += (file + "<|endoftext|>") 

        #Set class variables
        self.n_positions    = n_positions
        self.size           = 1024*1024#arbitrary

        #Start in warmup mode
        self.warmup         = True
        self.train_i        = .05    

        #Perform cleaning operations
        

    def print_stats(self):
       print(f"Created dataset:\n\t\t{len(self.text.split(' '))/1_000_000:.2f}M words\n\t\t{len(self.text)/1_000_000:.2f}M characters\n\t\t{len(self.tokenized_text)/1_000_000:.2f}M tokens\n\t\t{len(set(self.text))} unique chars\n\n")


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
        self.text           = self.text.lower()
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
            self.text   = self.text.replace(x,y)

        # while "  " in self.text:
        #     self.text       = self.text.replace('  ',' ')

        #display current vocab 
        
        # with open("bigdata.txt",'w',encoding='utf_8') as writefile:
        #     writefile.write(self.text)
        

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



#Create a ASCII version of the transcripts 
# also take out stops words and other stupid stuff
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

if __name__ == "__main__":
    
    
    #get_yt_captions("ytdump.txt")

    resave()