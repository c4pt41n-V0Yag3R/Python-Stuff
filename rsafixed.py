import rsa
(pubkey, privkey) = rsa.newkeys(512)
msg = 'a'.encode('utf8')
crypto = rsa.encrypt(msg, pubkey)
print("Encrypted: "+str(crypto))
print("Decrypted: "+str(rsa.decrypt(crypto, privkey)))
