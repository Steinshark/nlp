import re
import urllib.request 
import requests
import unidecode 
import urllib

ascii_cleaner   ={
        "\U0001f3b8":" ", "\u0171":"u","\u202a":" ",

        '\u0131':'i','\xa3':'L',
        '\xa0':' ','\u2019':"'",
        '\xb0':'degrees','\u266a':'[music]','\u2026':'...',
        '\U0001f60a':':)', "\x2018":"'",
        "\u2018":"'",        "\xe9":"e",
        "\u20ac":'Euro',        "\u2044":"/",
        "\xe8":'e',        "\xf1":"n",
        "\u201c":'"',        "\u201d":'"',
        "\xed":'i',        "\xc5":'A',
        "\xf3":'o',        "\u2014":'-',
        " um ": " ",        " uh ": " ",
        " i i ": " i ",        'ç':"c",
        'с':"c",        'π':"pi",
        '≅':"~=",         '™':"TM",
        '𝑛':"n",          '𝑃':"P", 
        'ℙ':"P",         '−':"-", 
        '²':"^2",         '𝜋':"pi", 
        '𝐸':"E",         '~':"~",         'γ':"gamma",         '′':"`", 
        '¹':"^1",         '⁵':"^5", 
        'в':"B",         '𝐺':"G",         '₂':"_2",         '∀':" for all ", 
        'м':"M",        '∃':" there exists ",        'Δ':"Delta",         '𝜃':"Theta", 
        '‽':"?!",         '𝛿':"sigma",         'ő':"o",         '𝐻':"H", 
        '�':"?",        '∈':" element of ",
        '₁':"_1",        'δ':"sigma",         '∎':"[]",         '⊗':"X",         'ɸ':"phi",         'ν':"v",         'ℕ':"N", 
        '\u2009':"?",        '𝐷':"D",         '·':" dot ",        'ä':"a",        '̶':"?", 
        '⁰':"degrees",         'É':"E",        'à':"a",        'е':"e",        'д':"D",        '×':"x",
        '→':"->",         'ö':"o",        'Ο':"O",        '𝐶':"C",
        '𝑎':"alpha",        'ú':"u",        'т':"T",        '𝐹':"F",        '½':"1/2",        'ℝ':"R", 
        'θ':"Theta",         'έ':"e", 
        'ô':"o",         '³':"^3",         'á':'a',         '𝓁':"l", 
        '´':"`",         'ń':"n",         '⅓':"1/3",         'ï':"l", 
        '･':"dot",         '–':"-",         '𝐵':"B",         '∩':"intersection", 
        '𝑏':"b",         '∞':"infinity",         '∂':"b",         '¡':"!",         'ü':"u", 
        '⁴':"^4",         'ᵢ':"_i", 
        '♫':"[music]",         'υ':"v",         '😲':":)",        'ë':"e",        'ã':"",
        'ā':"a",        'š':"s",        'ř':"r",        "ō":"o",        "õ":"o",        'й':"n",
        'ì':"i",        'ī':"i",        'Š':"S",        'ù':"u",        '鼎':"?",'Н':"?",'у':"?",
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


def get_transcript(video_html:str):

    #Find url 
    try:
        pre_parsed_url  = re.findall('(https://www.youtube.com/api/timedtext.{1,400}lang=en)',video_html)[0]
    except IndexError:
        #Could indicate its not in English (it sucks)
        with open("dump.html",'w',encoding='utf_8') as writefile:
            writefile.write(video_html)
        print("out")
        raise IndexError
    parsed_url      = pre_parsed_url.replace(r"\u0026","&")

    response        = requests.get(parsed_url,timeout=3,proxies=urllib.request.getproxies())

    transcript      = ""
    for line in response.text.replace("text start=","\n").split("\n")[1:]:
        line = line.split(">")[1].split("<")[0].replace(r"&amp;#39;","'")
        transcript += line + "\n"

    transcript = unidecode.unidecode(transcript)
    # for key,val in ascii_cleaner.items():
    #     transcript = transcript.replace(key,val)

    # transcript  = transcript.encode('ascii','replace').decode()

    return transcript.replace("\n"," ")


def filter_bad_content(video_transcript:str):

    #Make determination on only 20_000 charas 
    decision_window     = video_transcript[:10_000].lower()
    decision_len        = len(decision_window)

    #Reject anything less than 500 chars 
    if decision_len < 500:
        return False 
    


    #Make checks for bad keywords 
    #Reject if font occurs more than 200 times 
    if decision_window.count("font") > 200:
        return False 
    elif (decision_window.count("[music]")*len("[music]"))/decision_len > .1:
        return False 
    elif (decision_window.count("[applause]")*len("[applause]"))/decision_len > .1:
        return False

    return True


def is_transcribed(video_transcript:str):

    #Get ratio of "." to len 
    punct_count = video_transcript.count(".")
    tot_len     = len(video_transcript)

    return ((punct_count / tot_len) > .005)

if __name__ == '__main__':

    html    = open('ex.html','r',encoding='utf_8').read()
    get_transcript(html)