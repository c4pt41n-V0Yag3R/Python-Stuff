import os
def testfunc(search_term , PATH = 'C://'):
    '''
    search in the whole pc or specified 'PATH' (e.g. 'c://') and its subdirectories for the search term
    and printing the absolute path of matched results.
    '''
    works = False
    for folder, subfolders, files in os.walk(PATH):
        path = os.path.abspath(folder)
        for file in files:
            if search_term.lower() in file.lower():
                filePath = os.path.join(path, file)
                works = True
    if not works:
        print('No match for the term: {} in drive {}'.format(search_term.lower(),PATH))
    else:
        print('Search for tstwrd Successful!!!'.format(search_term.lower()))
    return works
def search_here(search_term):
    '''
    search in the current directory and subdirectories for the search term
    and printing the absolute path of matched results.
    '''
    paths = []
    result = 0
    for folder, subfolders, files in os.walk(os.getcwd()):
        path = os.path.abspath(folder)
        for file in files:
            if search_term.lower() in file.lower():
                filePath = os.path.join(path, file)
                result +=1
                paths.append(filePath)
                print(str(result)+". "+filePath)
                print()
    if result==0:
        print('No match for the term: {} in "{}" was found.'.format(search_term.lower(),os.getcwd()))
    else:
        print('{} results containing "{}" in "{}" was found.'.format(result,search_term.lower(),os.getcwd()))
        i = input('Type a number to explore repective file.\n>>')
        if i.lower() == 'quit':
            quit()
        else:
            os.system("start "+paths[int(i)-1])
    
def search_everywhere(search_term , PATH = 'C://'):
    '''
    search in the whole pc or specified 'PATH' (e.g. 'c://') and its subdirectories for the search term
    and printing the absolute path of matched results.
    '''
    paths = []
    result = 0
    for folder, subfolders, files in os.walk(PATH):
        path = os.path.abspath(folder)
        for file in files:
            if search_term.lower() in file.lower():
                filePath = os.path.join(path, file)
                result +=1
                works = True
                paths.append(filePath)
                print(str(result)+". "+filePath)
                print()
    if result==0:
        print('No match for the term: {} in drive {}'.format(search_term.lower(),PATH))
    else:
        print('\n\n{} results containing "{}" in drive {} was found.'.format(result,search_term.lower(),PATH))
        i = input('Type a number to explore repective file.\n>>> ')
        if i.lower() == 'quit':
            quit()
        else:
            os.system("start "+paths[int(i)-1])
        
        
        
###################### test:        
def test():
    with open('jhello.py','w') as f: # just a test file. you can test it on your pc with any
        f.write('a test file to see if my search works or not') # term you want! 
    testfunc('jhello') # search locally and in the directory of this python script
    os.remove('jhello.py')
a = input("Search files for kwrd: ")
search_everywhere(a)
 
# search all the folders in your pc (Not gonna work on sololearn) 

#search_everywhere('HellO' , PATH = 'F://')
