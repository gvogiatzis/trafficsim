#!/bin/zsh
rm *.csv
touch cols.csv

for i in {1..10}
do
	python -u bin/trafficrl.py $* | tee /dev/tty | awk 'BEGIN {FS="="} /Training/ {printf("%f\n", $2); fflush()}' > tmp1.csv
	if (($i>1)); then
		paste -d "," cols.csv tmp1.csv > tmp2.csv
	else
		cp tmp1.csv tmp2.csv
	fi	
	mv tmp2.csv cols.csv
	rm tmp1.csv
	python utils/pltavg.py -s 10 -t "Run $i of 10" -x "episode" -y "reward" < cols.csv | /Users/george/.iterm2/imgcat -W 50%
done
