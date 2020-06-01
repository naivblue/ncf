# NCF
Neural collaborative filtering(NCF)는 nueral networks를 사용하여 user-item interaction 데이터를 학습하는 deep learning based framework 이다. 해당 코드는 Multi-Layer Percenptron(MLP)으로 학습한다.


### data
데이터는 임의의 데이터를 사용할 수 있으며, ./data/ 아래에 위치시키면 된다.

### Run
--use_cuda를 명시하면 gpu에서 실행되고, 명시하지 않으면 gpu를 사용하지 않고 실행된다. 

data_path 와 학습이 완료된 모델의 model_path는 반드시 명시해 주어야 한다.
```shell script
python ./train.py --use_cuda --data_path ./data --model_path ./final_model
``` 
