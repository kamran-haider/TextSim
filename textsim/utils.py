"""
Utility functions
"""
import re
import string
from unidecode import unidecode


def text_cleaner(text, collect_emojis=False, remove_punctuation=False):
    """
    Cleans text of unwanted characters.

    Parameters
    ----------
    text

    Returns
    -------

    """
    # remove non-unicode characters and convert to lower case
    text = unidecode(text.lower())
    # remove html tags and links
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

    # collect emojis and put at the end of the document
    if collect_emojis:
        emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        emojis = " ".join(emojis).replace('-', '')
        text = text + " " + emojis

    # remove punctuation marks
    if remove_punctuation:
        punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))
        text = punct_re.sub(' ', text)

    return text

