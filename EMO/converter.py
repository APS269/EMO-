# from googletrans import Translator

# def hinglish_to_english(text):
#     translator = Translator()
#     translated_text = translator.translate(text, src='hi', dest='en').text
#     return translated_text

# from googletrans import Translator

# def hinglish_to_english(text):
#     translator = Translator()
#     translated = translator.translate(text, src='hi', dest='en')
#     translated_text = translated.text
#     return translated_text

from googletrans import Translator

def hinglish_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='hi', dest='en')
    
    if isinstance(translated, list):
        translated_text = translated[0].text
    else:
        translated_text = translated.text
    
    return translated_text
