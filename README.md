
# chungnam_chatbot

- - -
### What is Pytorch?
Lua 언어로 개발되었던 딥러닝 프래입워크
* 단순함
* 성능 функциональность
* 직관적인 interface

### 개념
* GPU - 연산 속도를 빠르게 하는 역할을 한다.
* Tensor - 데이터 형태
* 동적 신경망 (dynamic neural network) - 훌련은 반복할 때마다 네트워크 변경이 가능한 신경망

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






```python
import time
def main():
    print('this is test')
