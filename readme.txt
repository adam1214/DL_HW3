*  安裝numpy, matplotlib, opencv2, os, random, argparse套件
*  直接執行main.py即可訓練模型與測試模型的準確率，main.py有參數設定介面可使用，使用方式如下:

usage: main.py [-h] [-l LR] [-e EPOCH] [-b BATCH_SIZE]
optional arguments:
-h, 		 --help            		show this help message and exit
-l LR, 	 	 --lr LR        		learning rate
-e EPOCH, 	 --epoch EPOCH			total training epoch
-b BATCH_SIZE,   --batch_size BATCH_SIZE	batch size for training & testing

*  預設的超參數設定可以讓testing accuracy達到98.8 %