#!/usr/bin/python
import MyRsa
#IT WORKS!!!
pubkey, privkey = MyRsa.make_key_pair(8)
message = input("Message to encrypt:\n>>>")
newm = ""
for i in message:
    c = ord(i)
    newm += chr(pubkey.encrypt(c))
print("Encrypting....\nYour encrypted message: "+newm)
print("Decrypting...")
decmess = ""
for i in newm:
    c = ord(i)
    decmess += chr(privkey.decrypt(c))
print("Your decrypted message: "+decmess)
print(MyRsa.PublicKey)
