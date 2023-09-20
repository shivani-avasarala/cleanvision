"""
WIKI-RELATED FUNCTIONS
"""

import requests
import urllib
import json
import re

# DO NOT CHANGE
QUERY_STR_PRE_ARTIST = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles="
QUERY_STR_POST_ARTIST = "&exintro=1"


def get_wiki_extract(artist_str):
    """
    Returns a JSON containing page information for a specific Wiki file.
    Needs a string with the artist's name.
    """
    url = QUERY_STR_PRE_ARTIST + make_spaces_ascii(artist_str) + QUERY_STR_POST_ARTIST
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


def make_spaces_ascii(artist_str):
    """
    Replaces all whitespace characters and replaces them with their
    ASCII variant, allowing for a query search.
    (just space, not tab or newline)
    """
    while artist_str.find(' ') != -1:
        artist_str = artist_str.replace(' ', '%20')
    return artist_str


def pull_extract(wiki_json):
    """
    Takes the JSON scrubbed from the Wiki article and returns some
    summary text as a string.
    """
    json_dict = eval(json.dumps(wiki_json))
    if "query" in json_dict.keys():
        query = json_dict.get("query")
        pages = query.get("pages")
        page_id = pages.get(list(pages.keys())[0])
        extract = page_id.get("extract")
        return extract
    else:
        print("Potential issue with the Wiki article.")
        return ""   # nothing for now

def remove_html_tags(text):
    """ Remove html tags from a string
        Credit: https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# def remove_newlines(text):
#     clean = re.compile('\n')
#     return re.sub(clean, '', text)

# def main():
#     # TEST 1
#     test = get_wiki_extract("Freddie Mercury")  # 1 space
#     pull_extract(test)
#
#
# if __name__ == "__main__":
#     main()