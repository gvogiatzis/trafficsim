# python bin/trafficrl.py --seed 1234  --use-gui --gui-config-file sumo_data/RJ2/view.xml  --car-length 50 --vehicle-spawn-rate 0.1 --real-routes-file output/converter/real_routes.pk train --net sumo_data/RJ2/RJ2.net.xml

python bin/trafficrl.py --seed 1235 --save-tracks --vehicle-spawn-rate 0.1 --num-episodes 5 --episode-length 50 --use-gui --step-length 0.1 --gui-config-file sumo_data/RJ2/view.xml  --record-screenshots --input output/models/model_final.pt  --car-length 80 --real-routes-file output/converter/real_routes.pk test --net sumo_data/RJ2/RJ2.net.xml


python bin/changescale.py -m 1:1920 -m 3:1920 -m 2:1080 -m 4:1080 --output-format "rescaled_frame%05d.txt" sumo_data/RJ2/boxes/frame*.txt

python bin/sumotoreal.py fitbbox --bbox-data-path "sumo_data/RJ2/boxes/res*.txt"

python bin/sumotoreal.py convertseq sumo_data/RJ2/background.png

python bin/changescale.py -d 1:1920 -d 3:1920 -d 2:1080 -d 4:1080 --output-format "rescaled_frame%05d.txt" output/sumo_to_real_txt/frame*.txt

rm -rf output/sumo_to_real_txt/frame*.txt

rename 's/rescaled_//' output/sumo_to_real_txt/*.txt

mkdir output/videos

ffmpeg -y -i output/sumo_screenshots/%09d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p output/videos/screenshots.mp4

ffmpeg -y -i output/sumo_to_real_figs/img%06d.png -pix_fmt yuv420p output/videos/tracks_with_bboxes.mp4

ffmpeg -y -i output/videos/screenshots.mp4 -vf scale=640:480 output/videos/screenshots_lr.mp4

ffmpeg -y -i output/videos/screenshots_lr.mp4 -i output/videos/tracks_with_bboxes.mp4 -filter_complex hstack output/videos/output.mp4

open output/videos/output.mp4
