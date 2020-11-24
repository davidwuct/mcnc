import os
from urllib.request import urlopen, URLError, HTTPError


def dlfile(url):
    # Open the url
    try:
        f = urlopen(url)
        print("downloading " + url)

        # Open our local file for writing
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())

    #handle errors
    except HTTPError as e:
        print ("HTTP Error:", e.code, url)
    except URLError as e:
        print ("URL Error:", e.reason, url)


def main():
    for i in range(1994, 2011):
        for j in range(1, 13):
            if i == 1994 and j < 7:
                continue
            year = str(i)
            month = str(j).zfill(2)
            url = ("http://150.135.174.192/data/xml/nyt_eng_%s%s.xml.gz" % (year, month))
            dlfile(url)

if __name__ == '__main__':
    main()