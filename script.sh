#!/bin/sh

#python bin/trafficrl.py --save-tracks --num-episodes 5 --episode-length 50 --use-gui --step-length 0.1 --gui-config-file sumo_data/RussianJunction/view.xml  --record-screenshots --input output/models/model_final.pt  --car-length 50 test --net sumo_data/RussianJunction/RussianJunctionWide.net.xml

python bin/sumotoreal.py fitbbox --bbox-data-path "boxes/res*.txt"

python bin/sumotoreal.py convertseq sumo_data/RussianJunction/image.png

ffmpeg -y -i output/sumo_screenshots/%09d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p output/videos/screenshots.mp4

ffmpeg -y -i output/sumo_to_real_figs/img%06d.png -pix_fmt yuv420p output/videos/tracks_with_bboxes.mp4

ffmpeg -y -i output/videos/screenshots.mp4 -vf scale=640:480 output/videos/screenshots_lr.mp4

ffmpeg -y -i output/videos/screenshots_lr.mp4 -i output/videos/tracks_with_bboxes.mp4 -filter_complex hstack output/videos/output.mp4

open output/videos/output.mp4
