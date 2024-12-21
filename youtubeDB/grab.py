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
from fake_headers import Headers
from fp.fp import FreeProxy
from multiprocessing import Pool




def generate_proxies_list(timeout=.5,timelim=5):
    proxies         = [] 
    t0              = time.time()
    while time.time() - t0 < timelim:
        proxies     += FreeProxy(timeout=timeout,https=False,country_id=['US']).get_proxy_list(False)
    
    return [{'http':proxy} for proxy in list(set(proxies))]


def parse_youtube_urls(html_text:str):
    yt_urls = [it[0] for it in re.findall(r"(watch\?v=([a-z]|[A-Z]|[0-9]|-|_)+)",html_text)]
    return yt_urls


def parse_stackoverflow_urls(html_text:str):
    so_urls     = [it[0] for it in re.findall(r"(/questions/[0-9]{6,9}/([a-z]+|-)+)",html_text)]
    return so_urls


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
            time.sleep(.02+random.random())
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
            with open(f"C:/data/nlp/urls/scraped_urls_{subject_start}{random.randint(100,999)}.json",'w',encoding='utf_8') as writefile:
                writefile.write(json.dumps(list(url_jackpot)))
            save_lim += 1_000

            if len(url_jackpot) > lim:
                print(f"exiting")
                break 


def crawl_so_for_urls(start_url="https://stackoverflow.com/questions/65557230/where-can-i-find-a-specific-dataset",savename=f"stackoverflow{random.randint(100,999)}",lim=20_000):
    url_pre         = "https://stackoverflow.com/"
    
    all_urls        = set()
    #Get list of all videos found 
    for fname in os.listdir("C:/data/nlp/urls"):
        if not "stackoverflow" in fname:
            continue
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
            time.sleep(.5+random.random()*3)
            headers         = Headers(headers=True)
            req             = requests.get(url=next_url,timeout=3,headers=headers.generate())

            #~10% of the time, print stats
            if random.random() < .1:
                print(f"fetched {len(url_jackpot)} urls")

            #Parse for URLs and add to set
            if req.status_code == 200:
                html        = req.text 
                
                #Save the html to dataset
                with open(os.path.join("C:/data/nlp/stackoverflow",str(random.randint(1_000_000_000,999_999_999_999))+".txt"),'w',encoding='utf_8') as writefile:
                    writefile.write(html)

                #Save the urls
                discoveries = parse_stackoverflow_urls(html)
                for url in discoveries:
                    
                    if not url in all_urls:
                        queued_urls.add(url_pre+url)
                        url_jackpot.add(url_pre+url)
            else:
                print(f"rec code {req.status_code}")
        #Ok on timeout
        except requests.Timeout:
            print(f"failed to fetch url {next_url}")
            queued_urls.add(next_url)
            time.sleep(random.randint(2,7))

        if len(url_jackpot) > save_lim or len(url_jackpot) > lim:
            print(f"saved {len(url_jackpot)} urls")
            with open(f"C:/data/nlp/urls/scraped_urls_{savename}{random.randint(100,999)}.json",'w',encoding='utf_8') as writefile:
                writefile.write(json.dumps(list(url_jackpot)))
            save_lim += 1_000

            if len(url_jackpot) > lim:
                print(f"exiting")
                break 


def download_yt_transcripts(blank=None):
    all_urls        = set()
    headers         = Headers(headers=True)
    #Gather proxies
    proxies         = generate_proxies_list()
    print(f"generated {len(proxies)} proxies")
    #Get a set of all videos found 
    for fname in os.listdir("C:/data/nlp/urls"):
        if not "scraped_urls" in fname:
            continue
        fname   = os.path.join("C:/data/nlp/urls",fname)
        with open(fname,'r',encoding="utf_8") as readfile:
            for item in json.loads(readfile.read()):
                all_urls.add(item)
    print(f"found {len(all_urls)} existing urls")
    all_urls        = list(all_urls)
    random.shuffle(all_urls)
    all_urls        = set(all_urls)
    
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
            time.sleep(random.random()+.2)    
            html_request    = requests.get(url=next_url,timeout=3)#,proxies=random.choice(proxies))

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

                except IndexError as e:
                    print(f"index")
                    continue
            else:
                print(f"rec code {html_request.status_code}")
                if not html_request.status_code == 404:
                    time.sleep(random.randint(30,5*60))
        except requests.exceptions.ReadTimeout:
            print(f"read timeout")
            time.sleep(random.randint(30,5*60))
        except requests.exceptions.ConnectionError:
            print("connection error")
        except json.decoder.JSONDecodeError:
            pass


def crawl_developer_tech_for_urls(start_url="https://www.developer-tech.com/news/holistic-open-source-tools-counter-ai-development-risks/"):

    found_urls      = set()
    unsearched_urls = {start_url}
    next_step       = 100

    while unsearched_urls:
        next_url    = unsearched_urls.pop()

        try:
            urls    = utils.get_url_html(next_url)
            for url in urls:
                unsearched_urls.add(url)
                found_urls.add(url)
            print(len(found_urls))
        except ValueError as ew:
            print(f"got val error {ew}")
            pass 

        if len(found_urls) > next_step:
            print(f"found {len(unsearched_urls)} urls")
            next_step += 100
            with open("C:/data/nlp/developer-tech/urls.txt",'w') as writefile:
                writefile.write(json.dumps(list(found_urls)))

        

        
if __name__ == "__main__":

    command     = sys.argv[1]

    if command == "crawl":
        for url in ["https://www.youtube.com/watch?v=DVd6WRjeqMU","https://www.youtube.com/watch?v=zUHq8AWR1Rg","https://www.youtube.com/watch?v=DAANNGvMANM","https://www.youtube.com/watch?v=Jkft49A8pOk","https://www.youtube.com/watch?v=rSsaoNmVnZA","https://www.youtube.com/watch?v=R2tgByRCLzM","https://www.youtube.com/watch?v=eMlx5fFNoYc"]:

            crawl_yt_for_urls(start_url=url,subject_start=str(random.randint(100_000_000_000,999_000_000_000)),lim=40_000)
    elif command == 'stack':
        crawl_so_for_urls()
    elif command == 'download':
        download_yt_transcripts()
    elif command == 'news':
        crawl_developer_tech_for_urls()
            #download_yt_transcripts()