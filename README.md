# NCF
Neural Collaborative Filtering recommendation



### data
데이터는 임의의 데이터를 사용할 수 있으며, ./data/ 아래에 위치시키면 된다.

### Run
train epoch마다 evaluation이 같이 진행되며 평가지표는 Top-k hit ratio 와 ndcg이다.

--use_cuda를 명시하면 gpu에서 실행되고, 명시하지 않으면 gpu를 사용하지 않고 실행된다. 

data_path 와 학습이 완료된 모델의 model_path는 반드시 명시해 주어야 한다.
```shell script
python ./train.py --use_cuda --data_path ./data --model_path ./final_model
``` 
