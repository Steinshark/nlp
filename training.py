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

TOK_DB              = f"{PATH}/tokens"
TOK_DB_CLEAN        = f"{PATH}/tokens_clean"

PREV_RUNS           = f"{PATH}/prev"

MODELS              = f"{PATH}/models"

FINE                = f"{PATH}/fineweb"

FINEDB              = f"{PATH}/fine"

ULTRA               = f"{PATH}/ultrafine_tokens"

INTER               = f"{PATH}/whitelist"

for fpath in [CRAWL_DB,TOK_DB,PREV_RUNS,MODELS,FINE,FINEDB,TOK_DB_CLEAN,ULTRA,INTER]:
    if not os.path.exists(fpath):
        os.mkdir(fpath)

END_TOKEN           = "<|endoftext|>"

ALLOWABLE_CHAR      = string.ascii_lowercase + string.ascii_uppercase + "1234567890!@#$%^&*()~`':;{[}]_-+=<,>.?/}|\\ \n\t" + '"'

ENGL_DICT           = enchant.Dict("en_US")

LOWER               = False