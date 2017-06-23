import os

csv_file_name = "../webshot_data/popular-web-sites.csv"

with open(csv_file_name) as f:
    lines = f.readlines()
    for line in lines:
        path = line.split(',')[1]
        path = path.replace('\n','')
        if not path.startswith('http') and not path.startswith('www.'):
            path = 'www.' + path
        # end if
        sys_call = 'python webshots --width=1024 --height=768 '+ path
        print sys_call
        os.system(sys_call)
        mv_call = 'mv ' + path.replace('/','-') + '.1024x768.png ../webshot_data/' + line.split(',')[0] + '/'
        print mv_call
        os.system(mv_call)
        break
    # end for
#end with


