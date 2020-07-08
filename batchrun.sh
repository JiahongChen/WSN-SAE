python main.py  --dataset sst --num_compressed 100
sleep 5
python main.py  --dataset sst --num_compressed 200
sleep 5
python main.py  --dataset sst --num_compressed 302
sleep 5
python main.py  --dataset sst --num_compressed 400
sleep 5
python main.py  --dataset prec --num_compressed 50 --epoch 200
sleep 5
python main.py  --dataset prec --num_compressed 100 --epoch 200
sleep 5
python main.py  --dataset prec --num_compressed 206 --epoch 200
sleep 5
python main.py  --dataset prec --num_compressed 300 --epoch 200

