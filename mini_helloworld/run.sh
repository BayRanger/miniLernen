#!/bin/bash

gcc helloworld.c -o helloworld

#file basic info
printf "command: file helloworld \n output:\n "
file helloworld

printf "command: readelf helloworld \n output:\n "
readelf -h helloworld

# sudo ln -s /opt/homebrew/Cellar/binutils/2.42/bin/gobjdump objdump
#objdump -h helloworld

objdump -D helloworld