
for l in $(ls ../urls/*.txt)
do
mkdir -p ${l[@]:0:-4}
bash dw_util.sh $l $1 $2 #change 2 for number of threads
done
