#/bin/bash
xuehao="25121689"
mima="Mvh@5079"
calg=`curl -s http://10.10.42.3/md5calg|awk -F "\"" {'print $4'}`
v='2'
echo $calg
upass=`echo -n $v$mima$calg|md5sum|awk {'print $1'}`$calg$v
data="DDDDD="$xuehao"&upass="$upass"&R1=0&R2=1&para=00&hid1=&hid2=&0MKKey=123456"
echo $data

curl 'http://10.10.42.3/' -H 'Origin: http://10.10.42.3' -H 'Accept-Encoding: gzip, deflate' -H 'Accept-Language: en-GB,en-US;q=0.9,en;q=0.8' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (X11; Linux armv7l) AppleWebKit/537.36 (KHTML, like Gecko) Raspbian Chromium/65.0.3325.181 Chrome/65.0.3325.181 Safari/537.36' -H 'Content-Type: application/x-www-form-urlencoded' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8' -H 'Cache-Control: max-age=0' -H 'Referer: http://10.10.42.3/' -H 'Connection: keep-alive' --data $data --compressed

