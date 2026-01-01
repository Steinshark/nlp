import training 
import os 
import json 
import xxhash
from SimHash import simhash, get_stopwords
import time 

#This class maintains a DB of common crawl articles


class CCDB:


    def __init__(self):

        self.db_fp              = "D:/nlp/crawl docs/ccdb.json"

        self.docs               = {} 

        self.simhash_stopwords  = get_stopwords()

        self.load_db()


    def load_db(self):

        if not os.path.exists(self.db_fp):
            self.write_db()
            return 
        
        with open(self.db_fp,'r',encoding='utf-8') as readfile:

            self.docs               = json.loads(readfile.read())
        

    def write_db(self):

        with open(self.db_fp,'w',encoding='utf-8') as writefile:
            writefile.write(json.dumps(self.docs))


    #Get the unique id of a doc based on hash value 
    def get_doc_id(self,doc:str):

        usable_doc          = doc[:4096]
        doc_id              = str(xxhash.xxh3_64(usable_doc.encode()).hexdigest()) 

        return doc_id 


    #Used to compute the simhash of the document 
    def get_doc_simhash(self,doc:str):
        
        simhash_output  = simhash(doc,self.simhash_stopwords)
        try:
            simhash_output  = list(simhash_output)
        except TypeError:
            pass 
        
        return simhash_output


    #Add docs to DB id'd based on hash of first 4096 digits 
    def add_docs_to_db(self,doclist:list[str],save_every:int=10_000):

        for i,doc in enumerate(doclist):

            doc_id              = self.get_doc_id(doc)

            #In instances where we have a collission or duplicate, ignore  
            if doc_id in self.docs:
                print(f"skipping seen doc")
                continue 

            doc_simhash         = self.get_doc_simhash(doc) 
            self.docs[doc_id]   = {'contents':doc,'simhash':doc_simhash} 
        
            if i % save_every == 0:
                self.write_db()
                print(f"wrote db at {i}")


    def get_doc_by_id(self,doc_id:str):

        if doc_id in self.docs:
            return self.docs[doc_id]
        else:
            raise IndexError(f"Doc '{doc_id}' not in database")




def collect_docs(data_path:str):

    db                  = CCDB()

    text_file_paths     = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if not ".gz" in fname and ".wetc" in fname]
    write_docs          = [] 
    print(f"discovering {len(text_file_paths)} db files")
    time.sleep(10)

    for fpath in text_file_paths:
        docs            = open(fpath,'r',encoding='utf-8').read().split(training.END_TOKEN)

        for doc in docs:
            write_docs.append(doc)

    print(f"adding {len(write_docs)} documents")
    db.add_docs_to_db(write_docs)
    db.write_db()
    print(f"wrote {len(db.docs)} documents")


collect_docs("D:/nlp/crawl")