#!/bin/bash
for file in *.mp4; do
name=${file%%[.]*};
ffmpeg -i "$file" -t 00:00:10 -async 1 "../videos-short/$file"
done
