ROOT=${1:-../text_rank}

for file in `find $ROOT -name '*.py'`; do
    black -l 120 -t py36 $file;
done
