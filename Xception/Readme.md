# Xception 논문 리뷰

## “Deep Learning with Depthwise Separable Convolutions”

- Inception 모듈을 변형하여 parameter 수 유지하면서 성능 향상
- Cross-channel correlations와 spatial correlations를 완전 독립

---

## 1. Main idea

→ ‘Inception 모듈의 개념을 극대화하여 **채널과 공간을 완전히 분리해서 학습해보자**’

- 일반 CNN: 채널, 공간을 분리 없이 한 번에 학습
- **Inception**: 채널, 공간을 ‘부분적으로’ 분리
    - (1X1)로 채널 간 혼합 및 축소 → 3~4개 segments로 나눠 (kXk)로 공간 정보 학습
- **Xception**: 채널, 공간을 ‘완전히’ 분리
    - 채널별 (kXk)로 공간만 학습 → (1X1) 한 번으로 혼합
    - 중간에 비선형 활성화 함수 없고 residual 구조 적용 등
    - 이전에 사용하던 Aux_loss는 제거
    
- Figure로 Main idea 확인
    - 기존 Inception V3를 단순화하면 다음과 같이 그릴 수 있다.
        
        <img width="457" height="281" alt="image" src="https://github.com/user-attachments/assets/b0b6686c-f829-4a30-839e-e31b7cd88f5f" />

        
    - 이 1X1 Conv들을 한번에 묶어서 다시 그리면 다음과 같다.
        - 3~4개의 segments로 나뉘던 일반 Conv라서 segment 내부에서 채널이 다시 섞인다. → ‘완전 분리가 아니다’
        
        <img width="523" height="289" alt="image" src="https://github.com/user-attachments/assets/46ff8e67-e35b-4447-8c24-00b96eac539e" />

        
    - 이제 채널 수만큼 3X3을 통과하면 각각의 채널 내에서만 공간을 섞기 때문에 ‘완전한 분리’가 된다.
        
        (단, 실제 Xception은 순서가 다르고 차이점이 있긴 하다. 단지 Inception을 극대화한 것의 표현임)  
        
        <img width="505" height="275" alt="image" src="https://github.com/user-attachments/assets/0fa2fd2e-1278-4dd5-a2e0-dc1bf185e00b" />

        

## 2. Why Depthwise separable?

- **용어 정리**
    - **Depthwise**: (kXk), Channel 내 공간만 섞고(공간 특징 추출하고) 채널은 안섞는다.
    - **Pointwise**: (1X1), Channel 간 공간은 안섞고 채널만 섞는다.(채널 간 정보 혼합 및 차원 변환한다)
    - **`Depthwise Separable Conv`**: Depthwise(kXk) → Pointwise(1X1) 하는 것.
    - ‘separable’의 두 가지 의미: **Depthwise separable**(딥러닝의 DW→PW) vs **Spatially separable**(k×k → k×1 + 1×k)

1. 파라미터, FLOPs 대폭 감소 → 같은 크기에서 더 깊고 넓게
- 일반 (kXk) → $k^2C_{in}C_{out}$
- Depthwise separable → $k^2C_{in} + C_{in}C_{out}$
    
    Example)
    
     $k=3,\ C_{in}=32, \ C_{out}=64$
    
    일반: 18432개, Depthwise separable: 2336개 
    
    FLOPs도 거의 같은 비율로 감소한다. 
    
1. Spatial과 Channel의 역할분배
- Depthwise가 공간 패턴을 채널별로 먼저 뽑고, Pointwise가 그 결과를 채널 간 선형 결합으로 재조합하여 공간과 채널의 의미가 명확히 나뉘게 된다.
    
    ⇒ 성격이 다른 두 정보가 섞이지 않는다는 것이 모델의 판단 기준이 더 선명해진다?
    

## 3. The Xception architecture

- Input size=(3, 299, 299)
- Middle flow는 8번 반복
- 총 36 conv layers

<img width="1083" height="739" alt="image" src="https://github.com/user-attachments/assets/c9110311-dad7-46b6-a85c-9077702a93e5" />


- `SeparableConv 728, 3X3`: Depthwise 3X3 → BN → Pointwise 1X1(=출력 728) → BN (ReLU가 없다)
- Entry flow 초반 두 층은 일반 Conv Layer
    
    <img width="525" height="450" alt="image" src="https://github.com/user-attachments/assets/f24654af-f822-4a1e-b0b9-42b63be45524" />

    
- ReLU는 블록 전/후에 따로 들어가 있다.
- 처음과 마지막 블록을 제외한 모든 block에서 skip-connection을 사용
    
    <img width="523" height="419" alt="image" src="https://github.com/user-attachments/assets/5200b3e3-6542-4154-b95d-32afc4a04db9" />

    
- 마지막은 GAP

## 4. Evaluation

<img width="492" height="149" alt="image" src="https://github.com/user-attachments/assets/c5b189a0-2bc5-485b-af7e-147acfcff288" />


<img width="505" height="410" alt="image" src="https://github.com/user-attachments/assets/aa317fa3-2460-4691-af47-f5c4bbdca244" />



- 기존 Inception 모델 대비 단순하고 효율적이면서도 정확도 향상
