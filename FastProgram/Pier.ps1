$files = (ls IMPS**.txt).name

foreach ($file in $files){
python fastDTA.py $file
}
