import os 
import re 
import json
import urllib.request 
import requests 
import random 
import time 
import utils
import sys 
from utils import filter_bad_content, is_transcribed
import urllib

def parse_youtube_urls(html_text:str):
    yt_urls = [it[0] for it in re.findall(r"(watch\?v=([a-z]|[A-Z]|[0-9]|-|_)+)",html_text)]
    return yt_urls


def crawl_yt_for_urls(start_url="https://www.youtube.com/watch?v=MtGUr21H7UE",subject_start="chess1",lim=20_000):

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

        #Try to get the html text via requests
        try:

            #Wait randomly to avoid overly burdening server with requests
            time.sleep(.02+random.random()*2)
            req             = requests.get(url=next_url,timeout=3)

            #~10% of the time, print stats
            if random.random() < .1:
                print(f"fetched {len(url_jackpot)} urls")

            #Parse for URLs and add to set
            if req.status_code == 200:
                html        = req.text 
                discoveries = parse_youtube_urls(html)
                for url in discoveries:
                    
                    if not url in all_urls:
                        queued_urls.add("https://www.youtube.com/"+url)
                        url_jackpot.add("https://www.youtube.com/"+url)

        #Ok on timeout
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


def download_yt_transcripts():
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
        next_url        = all_urls.pop()
        next_id         = next_url.replace("https://www.youtube.com/watch?v=","")
        transcript_path = os.path.join("C:/data/nlp/yt_ascii",next_id+".txt")

        #Skip if we have already transcribed video 
        if os.path.exists(transcript_path):
            continue

        #Get html of video 
        try:
            #time.sleep(random.random()+.01)
            html_request    = requests.get(url=next_url,timeout=3,proxies=urllib.request.getproxies())

            if html_request.status_code == 200:

                #Get transcript
                try:
                    transcript  = utils.get_transcript(html_request.text)
                    #Get other data 
                    working_text    = html_request.text.split(',"videoDetails":{')[1].replace('&#39;',"'")

                    json_text       = '{'
                    count =          1
                    while count > 0:

                        next_char   = working_text[0]
                        working_text = working_text[1:]
                        json_text += next_char
                        if next_char == '}':
                            count -= 1
                        if next_char =='{':
                            count += 1
                    
                    video_data = json.loads(json_text)
                    video_data['transcript']    = transcript
                    video_data['is_quality']    = filter_bad_content(transcript)
                    video_data['transcribed']   = video_data['is_quality'] and is_transcribed(transcript)
                    with open(transcript_path,'w',encoding='ascii') as writefile:
                        writefile.write(json.dumps(video_data))


                except IndexError:
                    pass
        except requests.exceptions.ReadTimeout:
            time.sleep(random.randint(30,5*60))
        except requests.exceptions.ConnectionError:
            time.sleep(random.randint(30,5*60)) 
        except json.decoder.JSONDecodeError:
            pass


        
if __name__ == "__main__":

    command     = sys.argv[1]

    if command == "crawl":
        for url,subj in [["https://www.youtube.com/watch?v=ADfCseQjh2w","coffee3"],["https://www.youtube.com/watch?v=KuLUd1UIvVA","prime88"],["https://www.youtube.com/watch?v=sL8xBT2cc3c","chess33"],["https://www.youtube.com/watch?v=cTnV5RfhIjk","ben34"]]:

            crawl_yt_for_urls(start_url=url,subject_start=subj,lim=40_000)
    elif command == 'download':
        download_yt_transcripts()