import os


def main():
    
    with open('filelist.txt', 'r') as f:
        filelist = f.readlines()
    filelist = [f[:-1] for f in filelist]

    doc_ids = set([f[:-5] for f in os.listdir('.') if f[-5:] == '.json'])

    new_filelist = set()
    for filename in filelist:
        if filename[:-4] in doc_ids:
            continue
        new_filelist.add(filename)
    
    with open('filelist.txt', 'w') as f:
        for filename in new_filelist:
            f.write(filename + '\n')


if __name__ == '__main__':
    main()