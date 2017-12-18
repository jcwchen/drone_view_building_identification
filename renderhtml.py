import sys
import os



print('<!DOCTYPE html><html>')
head="""
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>
<script>
var index=0;
$(function(){

});

</script>
<style>
body{
    background-color:lightgrey;
    width: 960px;
    margin: 0 auto;
}
h1   {color:blue;}
.circle   {position: absolute; width: 5px; height: 5px; background: red;}
#annotation{
    color: red;
}
#frame{
    width: 800px;
}
</style>
</head>
<body>
"""

print(head)
#cml24

try:
    dir_name = sys.argv[1]
except:
    dir_name = "output/"

DIR = "/tmp3/jacky82226/baseline/"
DIR = os.path.join(DIR,dir_name)

demo = "http://cml24.csie.ntu.edu.tw:"+sys.argv[2]+"/"

file_length=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])



frame_dir = "frame"
poi_dir = "poi"

index = 0

for img_name in os.listdir(frame_dir):
    img_name = img_name.replace(".jpg","").replace(".png","")
    with open(os.path.join(poi_dir, img_name + ".txt"), 'r') as f:
        for line in f:
            token = line.strip().split("\t")
            gps_pos = [0, 0]
            [name, gps_pos[0], gps_pos[1], google_type, img_ref, gt] = token
            str_gps_pos = [gps_pos[0], gps_pos[1]]
            query_name = str(str_gps_pos[0]) + '_' + str(str_gps_pos[1])


            print("<div>")
            print(name)
            # query
            src=os.path.join(demo, 'search', query_name + ".jpg")
            img = "<img height=\'250\' src=\'"+src+"\'/>"
            print(img)
            # result
            src=os.path.join(demo, dir_name,"{}.jpg".format(index+1))
            img = "<img width=\'960\' src=\'"+src+"\'/>"
            print(img)
            print("</div>")

            index += 1
"""
for i in xrange(index):
    print("<div>")
    src=os.path.join(demo,"{}.jpg".format(i+1))
    img = "<img width=\'960\' src=\'"+src+"\'/>"
    print(img)
    print("</div>")
print('</body></html>')

"""
  
#for i in range(0 ,len(files), 3):
#    print '<a onclick="f1(\'' + "car" + str(i) + '\')"><img id=\'car' + str(i) + '\' src=\''  + server_url + parent_dir + files[i] + '\' width=\'130\' height=\'70\'></a>'

