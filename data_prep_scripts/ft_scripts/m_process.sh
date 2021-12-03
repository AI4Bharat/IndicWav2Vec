for i in $(ls "$1")
do
python m_prep_script.py "$1/$i" &
done
