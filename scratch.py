import random 


fname           = "ytdump.html"

contents        = open(fname,'r',encoding='utf_8').read()
replacers       = ['"4" lockup="true" is-slim-grid=""><!--css-build:shady--><!--css-build:shady--><div id="content" class="style-scope ytd-rich-item-renderer"><ytd-rich-grid-media class="style-scope ytd-rich-item-renderer" lockup="true" mini-mode=""><!--css-build:shady--><!--css-build:shady--><div id="dismissible" class="style-scope ytd-rich-grid-media"><div id="thumbnail" class="style-scope ytd-rich-grid-media"><ytd-thumbnail rich-grid-thumbnail="" use-hovered-property="" width="9999" class="style']

for replacer in replacers:
    if not "watch?v=" in replacer:
        contents     = contents.replace(replacer,'')

while "\n\n" in contents:
    contents = contents.replace("\n\n","\n")
writefile       = open(fname,'w',encoding='utf_8')
writefile.write(contents)