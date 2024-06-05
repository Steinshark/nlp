// Example program
#include <iostream>
#include <string>
#include <map>
#include <filesystem>
#include <fstream>

using namespace std;
int main(){  
    map<int,string> encoder;
    map<string,int> decoder;
  
    //Iterate through folder and add all text to global text
    string path         = "yt_captions2";

    string alltext     = "";

    for(const auto& fname : std::filesystem::directory_iterator(path)){
        //cout << fname.path() << "\n";

        ifstream readfile(fname.path());

        string line;
        while (getline(readfile,line)){
            alltext += line;
            
        }
    }

    //Make it a list
    cout << alltext.length() << " characters of text\n";
    string* texts       = new string[alltext.length()];

    for(int i=0; i < alltext.length();i++){
        texts->insert(texts->end(),alltext.at(i));
        
    }

    int mm = 0;
    while(encoder.size() < 512){

        //Find top pair 
        map<string,int> pairs;
        for(int i=0; i+1 < texts->size();i++){
            string pair         = string() + texts->at(i) + texts->at(i+1);
            // pair                += texts->at(i);
            // pair                += texts->at(i+1);


            pairs[pair]         += 1;
            //cout << pair << " is now at " << pairs[pair] << endl;
            // auto pair_it        = pairs.find(pair);
            // if(pair_it != pairs.end()){
            // }
            // else{
            //     pairs[pair]     = 1;
            // }
        
        }
        int maxcount            = 0;
        string toppair;
        for (auto const[pair,count] : pairs){
            if(count > maxcount){
                maxcount = count;
                toppair = pair;
                //cout << "override " << count << endl;
            }
        }
        cout << "Top pair is " << toppair << " with " << maxcount << endl;

    }
  
  
}