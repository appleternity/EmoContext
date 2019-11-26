import requests
from bs4 import BeautifulSoup
import os
import json
import re
import codecs

# reference: http://www.unicode.org/emoji/charts/full-emoji-list.html

# notes:
# html1.html: https://www.iemoji.com/meanings-gallery/smileys-people
# html2.html: https://www.iemoji.com/meanings-gallery/animals-nature
# html3.html: https://www.iemoji.com/meanings-gallery/food-drink
# html4.html: https://www.iemoji.com/meanings-gallery/symbols


def get_emoji_info(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.select(".codetable")
    return table

def get_list():
    # url = "https://www.iemoji.com/meanings-gallery/smileys-people"
    # res = requests.get(url)

    # with open("temp.html", 'w', encoding="utf-8") as outfile:
    #     outfile.write(res.text)
    filename = "html4.html"

    with open(filename, 'r', encoding='utf-8') as infile:
        html = infile.read()
    soup = BeautifulSoup(html, "html.parser")
    res = soup.select(".thumbnail a")

    with open("info.json", 'r', encoding='utf-8') as infile:
        data = [line for line in infile]
        info = json.loads(data[-1])
        index_offset = int(info["index"])

    with open("info.json", 'a', encoding='utf-8') as outfile:
        for index, r in enumerate(res, index_offset+1):
            print(index, len(res) + index_offset)
            url = "https://www.iemoji.com/{}".format(r.get("href"))
            info = get_emoji_info(url)
            text = json.dumps({"index":index, "info":str(info)})
            outfile.write(text+"\n")

def get_char(code):
    res = rf"\U{code:08X}"
    res = codecs.decode(res, "unicode_escape")
    return res

def parse_emoji_info():
    # Decimal Code Point(s)</td><td>128512</td>
    pattern = re.compile(r"Decimal Code Point\(s\)\<\/td\>\<td\>(?P<code>.+?)\<\/td\>")
    result = {}
    with open("info.json", 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            info = data["info"].replace("\n", "")
            res = pattern.search(info)
            if res == None:
                print(info)
                print(data)
                with open("char.json", 'w', encoding='utf-8') as outfile:
                    json.dump(result, outfile, indent=4)
                quit()

            res = res.groups()[0]
            char_res = ""
            for code in res.split(", "):
                code = int(code)
                char = get_char(code)
                char_res += char
            result[data["index"]] = {
                "index":data["index"],
                "code":code,
                "character":char_res
            }

    with open("char.json", 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, indent=4)

def main():
    # get_list()
    parse_emoji_info()

if __name__ == "__main__":
    main()