if [ ! -d "shakespear_data" ] || [ ! "$(ls -A shakespear_data)" ]; then
    if [ ! -d "data" ]; then
        mkdir data
    fi

    if [ ! -f data/raw_data.txt ]; then
        echo "------------------------------"
        echo "retrieving raw data"
        cd data || exit

        wget http://www.gutenberg.org/files/100/old/old/1994-01-100.zip
        unzip 1994-01-100.zip
        rm 1994-01-100.zip
        mv 100.txt raw_data.txt

        cd ../
    fi
fi

if [ ! -d "data/by_play_and_character" ]; then
   echo "------------------------------"
   echo "dividing txt data between users"
   python3 /datasets/text/shakespeare/preprocess_shakespeare.py data/raw_data.txt data/
fi