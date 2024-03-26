def upperfyShapespear() -> None:

    file = open('shakespeardata.txt')
    content = file.read()
    file.close()

    en_content = upperify(content)
    with open('shakespear_upper.txt', 'w') as file:
        file.write(en_content)
    file.close()

def createMorseShakespear() -> None:

    file = open('shakespear_upper.txt')
    content = file.read()
    file.close()

    en_content = encrypt(content)
    with open('shakespear_encrypted.txt', 'w') as file:
        file.write(en_content)
    file.close()


#Huffman encoding
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', '0': '-----',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--', '/': '-..-.', '(': '-.--.',
    ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-',
    '_': '..--.-', '"': '.-..-.', '$': '...-..-', '@': '.--.-.', ' ': '/', '\n' : '//'
}

def upperify(text):
    upperfied_text = ''
    for char in text.upper():
        if char in MORSE_CODE_DICT:
            upperfied_text += char.upper()
        else: #If not in the scheme, simply remove it
            upperfied_text += ""
    return upperfied_text

def encrypt(text):
    encrypted_text = ''
    for char in text.upper():
        if char in MORSE_CODE_DICT:
            encrypted_text += MORSE_CODE_DICT[char] + ' '
        else: #If not in the scheme, simply remove it
            encrypted_text += ""
    return encrypted_text

def decrypt(morse_code):
    morse_code = morse_code.split(' ')
    decrypted_text = ''
    for code in morse_code:
        for key, value in MORSE_CODE_DICT.items():
            if code == value:
                decrypted_text += key
    return decrypted_text

