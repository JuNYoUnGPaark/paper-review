# GoogLeNet 논문 리뷰

https://arxiv.org/pdf/1409.4842

## “Going deeper with convolutions”

- **다양한 size의 filter**(1X1, 3X3, 5X5)의 결과를 concat해서 사용
- **1X1 Conv로 차원을 축소**하여 parameter 수 감소
- **AUX Classifier**를 사용하여 vanishing gradient 완화
- VGGNet보다 더 적은 parameter수로 1등 달성

---

## 0. Why Inception?

- Network는 본질적으로 sparse 구조가 이상적 → 하드웨어 우호적인 밀집 연산으로 근사하자
- 여러 size의 kernels + 1X1 bottleneck 구조로 이를 구현했다.

## 1. Architectural Details

- Inception module: 다양한 kernel_size를 사용하여 결과를 concat
    
    <img width="633" height="289" alt="image" src="https://github.com/user-attachments/assets/3f6a6cd7-9784-4a8e-8759-38fb65f9a5e0" />

    
    - VGGNet은 3X3 filter를 두 번 통과하여 5X5 filter와 같은 receptive filed를 얻었지만, 해당 모델에서는 3가지를 모두 사용하여 얻은 feature map을 depth축으로 concat하는 방식을 채택한 것.
        
        → 행렬 크기가 같아야 concatenation이 가능하기에 **channel Size 조절이 핵심**
        
- **Inception Module with dimension reductions**
    
    <img width="602" height="309" alt="image" src="https://github.com/user-attachments/assets/f1892b0e-430d-4c70-b887-fec7d5541a6b" />

    - **1X1 conv filter가 하는 역할과 의미**
        1. BottleNeck 구조를 만들어 필요한 parameter 수 감소 → 차원 축소 
            
            - Example) 
            
            ---
            
            [1X1 Conv + 3X3 Conv]
            
            Input C = 192
            
            1X1 Conv = (96, 192, 1, 1) 
            
            3X3 Conv = (128, 96, 3, 3)
            
            ⇒ total params = (96X128X3X3) + (96X192X1X1) = 129,024개
            
            ---
            
            [only 3X3 Conv]
            
            Input C = 192
            
            3X3 Conv = (128, 192, 3, 3)
            
            ⇒ total params = 128X192X3X3 = 221,184개
            
            ---
            
            ⇒ result: **약 2배 차이 (1.71배)**
            
            ---
            
        2. 다른 Size의 filter와는 다르게 1X1 Conv는 주변의 *공간 정보를 섞지 않고* 오직 Channel 축으로만 재조합하여 새로운 feature map을 생성 
            - *“공간 정보를 섞지 않는다”*가 중요한 이유
                
                → 위치 정보는 그대로 보존한채 ‘무엇을 볼지’만 변화시킬 수 있다.
                

## 2. GoogLeNet Structure

<img width="1179" height="687" alt="image" src="https://github.com/user-attachments/assets/64a56bac-edf3-4a82-bb8b-bb86d1f09f7e" />


- 위에서 설명한 Inception Module이 뒤쪽 Layer까지 반복적으로 이어져있는 구조로 이루어져 있다.

<img width="1276" height="283" alt="image" src="https://github.com/user-attachments/assets/ddd0049d-b494-407b-a83e-db0fa85a8ab5" />


- One Inception Module

<img width="725" height="1116" alt="image" src="https://github.com/user-attachments/assets/c5598425-ad54-4b56-974d-d8e897fccf67" />


- (S)와 (V) 해석 방법
    
    ⇒ Input_shape=(224, 224) → 7X7 Conv를 통과 → size는 절반(표 참고)이 되어야 하므로 padding=2로 결정 (계속 이와 같은 방식)
    
- Example)
    
    Input_shape=(8, 8)일 때 Size를 유지하기 위해서 필요한 각각의 padding
    
    <img width="1897" height="1000" alt="image" src="https://github.com/user-attachments/assets/981eca66-1a3d-4a0d-9190-d685f8349d53" />

    

- **Auxiliary Classifiers**

<img width="242" height="571" alt="image" src="https://github.com/user-attachments/assets/d0d22815-3eca-418a-9a29-8e8b1699e131" />


- Inception(4a), (4d) output에 붙는다.
- Gradient Vanishing을 방지하기 위한 모듈
- SoftmaxActivation을 통과하여 최종 1000종 분류를 수행하고, Auxiliary Loss를 계산한다.
- 2개의 모듈이 존재하고 최종 loss는 다음과 같이 계산한다.

$$
Final \ Loss =out\ loss\\+\ 0.3\ *(aux\_loss1+aux\_loss2)
$$

- Train할 때만 사용하고 Test땐 제거한다.

## 3. Results

<img width="613" height="193" alt="image" src="https://github.com/user-attachments/assets/70ccf27c-7721-4fe9-a17e-08513d8aa30b" />


- 7개 모델을 (앞서 제시된 6개 + 더 큰 모델 1개) 앙상블
- 6개 모델은 구조부터 초기 웨이트까지 같고 data보는 순서만 다르게
- IMG 1장당 144장으로 augmentation하고 총 1008개 output에 대한 평균 계산하여 최종 분류 수행 (144 images * 7 models = 1008 outputs)
    
    <img width="1352" height="999" alt="image" src="https://github.com/user-attachments/assets/1a68c629-4efd-409e-938d-5eb2425999c3" />
    

## 4. Points of Confusion

### 1. 왜 Input에 가까운 layer가 vanishing gradient 문제가 심할까?

- 역전파 과정에서 해당하는 layer의 parameter들에 대한 편미분 값이 0에 가까워져 발생한다.
- 다음의 링크 참고: [[Vanishing Gradient ](https://www.notion.so/Vanishing-Gradient-1f6ccff627fb802bbd7dd94c801e6eeb?pvs=21)]

### 2. 왜 Auxiliary Classifiers를 사용하면 VG 완화가 될까?

- 직관적 설명: 모델이 깊어질 수록 앞단의 기울기 소실을 유발한다. 이를 완화하고자 중간 중간 손실값을 계산하고 이를 최종 손실값에 반영한다면 학습 과정에서 앞단의 영향력이 미미해지는 것을 막을 수 있다.
- 쉽게 말해 중간 중간 계산된 손실값들도 반영해서 업데이트 되기 때문에 **앞단을 비교적 덜 무시하며 학습**되는 것.

### 3. 왜 마지막에 GAP를 사용할까?

- feature map size = 7X7X1024로 flatten하면 수천만 개의 파라미터가 생긴다. 이에 **파라미터 수를 감소**시키고자 사용한다. (→ 과적합 방지)
- 각 Channel → Class로 가는 **linear layer를 상대적으로 얇게** 할 수 있다.
    
    (7X7 → GAP → 1X1) ⇒ linear(1X1X1000) 
    
    “거대한 FC=공간축까지 연결” ⇒ “작은 FC=채널만 연결” 
    
- 분류 전 최종 feature map에 대한 **요약 기능**을 한다. ㄴ

- input size가 달라도 사용이 가능해져 **이식성**이 좋아진다.
