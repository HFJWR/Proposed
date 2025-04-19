# RecSystems

## Running
For BLM Dataset  
~~~
python main.py --lr 0.05 --weight_decay 1e-6 --beta_class 0.90 --epoch 300 --patience 10 --dataset BLM
~~~

## Dataset
Format of train.txt val.txt test.txt: UserID,ItemID  
Format of item_category.txt: ItemID,CategoryID  