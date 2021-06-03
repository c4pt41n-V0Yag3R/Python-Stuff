#!usr/bin/python
#40.326645, -74.289894
#2017-09-02-65.06
#YYYY-MM-DD-<this day's DOW open>
import hashlib
from geopy import Nominatim
def geohash(datedow):
    geoloc = Nominatim()
    l = input("Address to encode: ")
    loc = geoloc.geocode(str(l))

    Y = input("Year: ")
    M = input("Month: ")
    D = input("Day: ")

    a = str(Y)+"-"+str(M)+"-"+str(D)+"-"+str(datedow)

    h = hashlib.new('ripemd160', b"\a")

    lat = loc.latitude
    long = loc.longitude

    frst = int(lat)
    thrd = int(long)
    print(frst)
    m = len(h.hexdigest())
    ha = str(h.hexdigest())

    s1 = ha[:int(m/2)]
    s2 = ha[int(m/2):]
    
    s1i = int(s1, 16)
    s2i = int(s2, 16)

    sec = s1i * 10**-len(str(s1i))-1
    fou = s2i * 10**-len(str(s1i))-1

    print(str(frst + sec) + " " + str(thrd + fou))
    
geohash(22334.07)
