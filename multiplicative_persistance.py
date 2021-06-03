def persistance(n, step):
    digits = [int(i) for i in str(n)]
    result = 1
    for j in digits:
        result *= j
    print(result)
    if(len(str(n))) == 1:
        print("DONE after ", step, " steps")
        return
    persistance(result,step+1)


num = input("Enter a number: ")

persistance(num,0)
