
import random
key = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()[]\{}|;\':",./<>?'
def encrypt(n, plaintext):
    """Encrypt text & return results."""
    res = ''

    for l in plaintext:
        try:
            i = (key.index(l) + n) % len(key)
            res += key[i]
        except ValueError:
            res += 1
    return res
def decrypt(n, ciphtext):
    """Decrypt string & return plaintext."""
    res = ''

    for l in ciphtext:
        try:
            i = (key.index(l) + n) % len(key)
            res += key[i]
        except ValueError:
            res += 1
    return res

print("===========================\nCaesar Cipher Decoder v2.0\nModded by JK\nOrig. idea by SOLOLEARN\n===========================")
text = input("Enter plaintext/ciphertext\n")
while True:
    ans = input("Encode(E) or Decode(D)?[q] to quit.\n")
    if ans.lower() == "encode" or ans.lower() == "e":
        k = input("Use personal key?(y/n)\n")
        if k.lower() == "y":
            while True:
                try:
                    e = input("Enter key(use an int)\n")
                    e = int(e)
                    if e > len(key):
                        e = e % len(key)
                    break
                except:
                    print("Use an integer.")
            print("Encrypted: "+encrypt(e, text))
        if k.lower() == "n":
            r = random.randrange(len(key))+1
            print("Your random encryption:\n"+encrypt(r, text))
    elif ans.lower() == "decode" or ans.lower() == "d":
        for offset in range(len(key)):
            print(str(offset)+". "+decrypt(offset, text)+"\n")
        input("Of these many decryptions, one is bound to be yours. Have fun!")
        break
    elif ans.lower() == 'q':
        quit()
    else:
        print("Choose to encode or decode, please.")
        
