def morse():
    morsdict = {'a': '.-', 'b': '-...', 'c':'-.-.', 'd': '-..', 'e':'.', 'f':'..-.', 'g':'--.', 'h':'....', 'i':'..', 'j':'.---', 'k':'-.-', 'l': '.-..', 'm': '--', 'n':'-.', 'o':'---', 'p':'.--.', 'q':'--.-', 'r':'-.-', 's':'...', 't':'-', 'u':'..-', 'v':'...-', 'w':'.--', 'x':'-..-', 'y':'-.--', 'z':'--..', '1':'.----', '2':'..---', '3':'...--', '4':'....-', '5': '.....', '6':'-....', '7':'--...','8':'---..','9':'----.', '0':'-----'}
    text = input("Enter text to turn to morse code:\n>>> ")
    text = text.lower()
    mors = []
    for i in list(text):
        for j in range(len(mors)):
            if mors[j] == None:
                mors.remove(mors[j])
                mors.insert(j+1, "|")
        mors.append(morsdict.get(str(i)))
        mors.append("/")
    mors.pop()
    morstring = ''.join(mors)
    print("Morse Code:\n"+morstring)
    input("Press any key to continue....\n")
morse()
