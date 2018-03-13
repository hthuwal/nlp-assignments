if [ $# -ne 2 ] 
then
	echo "2 Arguments Expected";
	exit 1;
fi

python run.py $1 $2