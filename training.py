import os 
import json
import string 
import enchant

PATH                = r"//STEINPC//S/nlp"

#Establish URL file
URL_PATH            = f"{PATH}/urls.json" 
if not os.path.exists(URL_PATH):
    with open(URL_PATH,'w') as writefile:
        writefile.write(json.dumps([]))

#Establish tracker file for already downloaded wet files 
DWNLD_PATH          = f"{PATH}/temp/downloaded_wet_files.json"
CRAWL_DB            = f"{PATH}/crawl"
PREV_RUNS           = f"{PATH}/prev"
MODELS              = f"{PATH}/models"
FINEWEB_BASE        = f"{PATH}/fineweb"
FINEWEB_CLEAN       = f"{PATH}/fineclean"
TRAINING_TEXT       = f"{PATH}/traintext"
TRAINING_TOKENS     = f"{PATH}/traintokens"


for fpath in [CRAWL_DB,PREV_RUNS,MODELS,FINEWEB_BASE,FINEWEB_CLEAN,TRAINING_TEXT,TRAINING_TOKENS]:
    if not os.path.exists(fpath):
        os.mkdir(fpath)

END_TOKEN           = "<|endoftext|>"
CODE_TOKEN          = "<|writecode|>"
RUNCODE_TOKEN       = "<|runcode|>"
WEB_TOKEN           = '<|websearch|>'
PROMPT_TOKEN        = '<|prompt|>'
RESPONSE_TOKEN      = '<|response|>'
RESERVE_1           = '<|reserve1|>'
RESERVE_2           = '<|reserve2|>'


SPECIAL_TOKENS      = [END_TOKEN,
                       CODE_TOKEN,
                       RUNCODE_TOKEN,
                       WEB_TOKEN,
                       PROMPT_TOKEN,
                       RESPONSE_TOKEN,
                       RESERVE_1,
                       RESERVE_2]


ALLOWABLE_CHAR      = string.ascii_lowercase + string.ascii_uppercase + "1234567890!@#$%^&*()~`':;{[}]_-+=<,>.?/}|\\ \n\t" + '"'


ENGL_DICT           = enchant.Dict("en_US")


LOWER               = False
