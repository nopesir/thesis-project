#!/bin/bash

# ---------------------------------------------------------------------- 
# Get the YOLOv4 trained weights on the Car class from 254300's Google Drive |
# ----------------------------------------------------------------------

gURL="https://drive.google.com/file/d/1Yz0KmzwIt_UnhIeoFHnd7qBQWhLQLlwo/view?usp=sharing"

# match more than 26 word characters  
ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

ggURL='https://drive.google.com/uc?export=download'

curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"


cmd='cd ../darknet/ && curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
echo -e "Downloading weights from GDrive...\n"
eval $cmd
