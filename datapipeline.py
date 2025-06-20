from tok import filter_by_topic, tokenize_corpus,load_tokenizer,train_tokenizer
from crawl import clean_fineweb
from training import *


if __name__ == '__main__':
    #Clean fineweb
    #clean_fineweb(min_score=.95)

    # #Perform whitelisting 
    #filter_by_topic()

    # #Tokenize based on new whitelist 
    FINAL_SIZE              = 32768
    RESERVED_TOK            = 4
    TOK_NAME                = f'tokenzier2'
    #train_tokenizer(FINAL_SIZE-RESERVED_TOK,TOK_NAME,db=INTER)

    # #Create final data
    tokenize_corpus(TOK_NAME,db=INTER,tok_db=ULTRA,n_workers=4)

