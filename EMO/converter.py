from googletrans import Translator

def hinglish_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='hi', dest='en').text
    return translated_text
