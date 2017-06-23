import requests
import sys
from lxml import html

#csv_file_name = sys.argv[1] # output file
csv_file_name = "../webshot_data/popular-web-sites.csv"
csv_file = open(csv_file_name, "w")


categories = ["Arts", "Business", "Computers", "Games", "Health", "Home", "Kids_and_Teens", "News", "Recreation", "Reference", "Regional", "Science", "Shopping", "Society", "Sports", "World"]
# categories = ["Adult", "Arts", "Business", "Computers", "Games", "Health", "Home", "Kids_and_Teens", "News", "Recreation", "Reference", "Regional", "Science", "Shopping", "Society", "Sports", "World"]

base = "http://www.alexa.com/topsites/category/Top/"

for category in categories:
    path = base + category
    print path
    r = requests.get(path)    
    tree = html.fromstring(r.content)    
    trs = tree.xpath('.//a/@href')
    for tr in trs:
        if tr.startswith( '/siteinfo/' ) :
            wp = tr.replace( '/siteinfo/', '' )
            if len(wp) > 1:
                print wp
                csv_file.write( category + ',' + wp + '\n')
    # end for
# end for

csv_file.close()
