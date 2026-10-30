import re

def split_description(description: str):
    """
    Split the description into a list of sentences.
    """
    return [caption.strip() for caption in description.splitlines() if caption.strip()]

def clean_caption(caption):

    caption= caption.lower().strip()
    caption= re.sub(r'[^a-z0-9 ]', '', caption)
    return " ".join(caption.split())


def clean_captions(captions):
    
    return [clean_caption(caption) for caption in captions]