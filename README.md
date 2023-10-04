- - - 
# chungnam_chatbot

- - -
#### 머신러닝 - 데이터 분류된 
* 지도 학습(knn, svm, 결정 트리, 로지스틱 회귀(분류들이 부드롭게 느립나다), 선형 회귀(단순하고 빠름)) 
* 비지도 학습(K -평균 군집화, 밀도 기반 군집 분석, PCA)
#### 딥러닝 - 정제되지 않은 데이터
* 합성 신경망 - 이미지 처리
    * 합성곱층
    * 풀링곱층
* 순환 신경망(RNN)
----
### What is Pytorch?
Lua 언어로 개발되었던 딥러닝 프래입워크
* 단순함
* 성능 функциональность
* 직관적인 interface

### 개념
* GPU - 연산 속도를 빠르게 하는 역할을 한다.
* Tensor - 데이터 형태
* 동적 신경망 (dynamic neural network) - 훌련은 반복할 때마다 네트워크 변경이 가능한 신경망
* layer(계층) - 모듈 또는 모듈을 구성하는 한 개의 계층으로 합성곱층, 선형계층
* 모듈 - 한 개 이상의 계층이 모여서 구성된 것
* 모델 - 최동적으로 원하는 네트워크

### Pytorch API
* torch - GPU를 지원하는 텐서 패키지
    * 다차원 텐서 기반으로 수학적 연산
* torch.nn - 신경망 구축 및 훈련 패키지
    * 합성곱 신경망, 순환 신경망, 정규화
* torch.multiprocessing 
    * 서로 다른 프로세스에서 동일한 데이터(텐서) 접근
* torch.utils: DataLoader
    * torch.utils.data.DataLoader - 모델에 데이터 제송하기 위한
    * torch.utils.bottleneck - 병목 현상을 디버깅하기 위한
    * torch.utils.checkpoint - 모델 또는 모델 일부 검사
* -------------------------------


### Pandas(판다스) - 데이터 호출 라이브러리 (JSON, PDF, CSV)
pip install pandas / 
import pandas as pd / 
import torch
```python
data = pd.read_csv(파일)

x / y = torch.from_numpy(data['x' / 'y'].values)unsqueeze(dim=1).float()
# csv 파일 x/y 컬럼의 값을 넘파이 배열로 받아 tensor(dtype)로 바꿈
```

### CustomDataset - 데이터 조금씩 나눠서 불러옴
* from torch.utils.data import Dataset
* from torch.utils.data import DataLoader
```python
class CustomDataset(Dataset):
    def __init__(self,csv_file): # 필요한 변수, 데이터셋 불러옴
        self.label = pd.read_csv(csv_file)
    def __len__(self): # 데이터셋의 길이, 즉, 총 샘플 수
        return len(self.label)
    def __getitem__(self, idx): #전체 x y 중에 해당 idx 데이터
        sample = torch.tensor(self.label.iloc[idx, 0:3]).int()
        label = torch.tensor(self.label.iloc[idx,3]).int()
        return sample, label
tensor_dataset = CustomDataset('file')
dataset = DataLoader(tensor_dataset, batch_size=4)


```
# Deep Learning
* 개념
    * 입력층 (input layer) - 더이터 받아드리는 층
    * 은닉층 (hidden layer) - 입력 노드부터 입력 값을 받아 가중합(weighted sum), 이 값을 활성화 함수에 출력층에 전달
    * 출력층 (output layer)
    * 가중치 (weight) - влияет на результат
    * bias 
    * 활성화 함수 (activation function) - 신호를 압력받아 이를 처리
        * 시그모이드 함수 
        * 렐루 한수 self.relu = torch.nn ReLu(inplace = True)
    * 손실 함수 (loss function)

---
### 합성곱 신경망 - convolutional neural network / CNN
이미지나 영상 처리
* 입력층
* 합성곱층 - 데이터에서 특성 추출하는 역할
* 풀링층(pooling layer) - 연산량 감소시키고 특성 벡터 추출하여 학습 효율 높게
    * 최대 풀링(max pooling) - 자주 사용함
    * 평균 풀링(average pooling)
* 출력층

### ARIMA 모델 - 자기 회귀와 이동 평균을 둘 다 고려하는 모형이다.
conda install -c conda-forge statmodels
pip install models
```python
model = Arima(series, order=(5,1,0)) # ARIMA function open
model_fit = model.fit(disp=0) # help to watch debug info but now display is False
print(model_fit.summary())
residuals = DataFrame(model_fit.resid) # error info in variable residuals in DataFrame
```


사전 훈련된 모델의 파라미터 학습 유무 지정 212p
```python
def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # 모델의 일부를 고정하고 나머지를 학습
```
### Clastering 
특성 비슷한 데이터끼리 묶어 주시는 머신 러닝 기법이다.
* it uses K-평균 군집화 (pip install kmeans-pytorch)
* 클러스터 개수를 편리하게 결정하기 위해 WCSS



# Flask
### Ip
An Internet Protocol address (IP address) is a numerical label such as 192.0.2.1 that is connected to a computer network that uses the Internet Protocol for communication.
### ip function
* it identifies the host, or more specifically its network interface
* it provides the location of the host in the network, and thus the capability of establishing a path to that host. 
* "A name indicates what we seek. An address indicates where it is. A route indicates how to get there.
