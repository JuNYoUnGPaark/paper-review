# CBAM 논문 리뷰

## “Convolutional Block Attention Module”

- Channel Attention만 하던 SE와는 다르게 **Spatial Attention까지 적용**하여 ‘무엇’과 ‘어디’를 볼 것인지 모델에 명시적으로 학습시킬 수 있었다.
- Average Pooling뿐만 아니라 **Max Pooling**까지도 활용했다.
- 기존의 CNN 구조에 쉽게 적용가능한 **높은 호환성**을 가졌다.

<img width="689" height="169" alt="image" src="https://github.com/user-attachments/assets/3b48724b-37c8-4dbb-beaa-12fede0f91f1" />



- 기존의 BAM 구조의 업그레이드 ver.

---

## 1. Why do we need ‘spatial attention’?

- SENet은 Channel Attention을 통해 feature map에 ‘무엇’을 더 집중할지를 알 수 있었다. [[SENet 논문 리뷰](https://www.notion.so/SENet-278ccff627fb80a8989ec89fb6edc4f2?pvs=21)]
- 이때 BAM은 Spatial Attention도 적용하여 ‘어디’를 더 집중할지도 파악할 수 있게 만들었다.

 → CBAM은 BAM과 마찬가지로 C, S Attention을 적용하고자 하는 Module이다. 

## 2. Structure

<img width="809" height="250" alt="image" src="https://github.com/user-attachments/assets/ce891dea-8019-45cb-954a-6893eedfefcc" />



- Channel Attention과 Spatial Attention이 직렬로 연결된 구조이다.
- 먼저 Channel Attention으로 ‘무엇’에 집중할 건지 학습 후 나온 중간 feature map에 다시 Spatial Attention을 통해 ‘어디’에 집중할 것인지를 학습한다.
    - (Channel → Spatial)
    - $F: Feature\ Map \ , \ F': after \ C.A \ , \ F'': after \ S.A$
    - $M_c,\ M_s: each \ module, \ ⊗: matrix\ mul$

$$
F'=M_c(F)⊗F, \\
F''=M_S(F')⊗F'
$$

- **Channel Attention Module (채널 별로)**
    
    <img width="773" height="191" alt="image" src="https://github.com/user-attachments/assets/7118505d-b3b0-4be7-9bfa-0af124eeed59" />


    
    - BAM은 Channel별로 Average Pooling만 했지만 CBAM은 Max Pooling까지 적용한다.
    - 정보를 압축한 표현은 Average Pooling이 가장 중요한 정보를 희석시키지 않은 표현은 Max Pooling이 확보한다는 생각.
    - 두 Pooling을 병렬 연산(1X1XC) → MLP → Element-wise sum. → Sigmoid(**1X1XC**) ⇒ Attention Score
        
        → 여기서 $MLP$는 AvgPool과 MaxPool에 공통 적용되는 동일한 신경망이므로 동일한 weight!
        
    
    $$
    M_c(F)= \sigma(MLP(AvgPool(F))+MLP(MaxPool(F)))\\
    =\sigma(W_1(W_0(F^c_{avg}))+W_1(W_0(F^c_{max})))
    $$
    
- **Spatial Attention Module (채널 축으로)**
    
    <img width="610" height="205" alt="image" src="https://github.com/user-attachments/assets/b7124aba-6f8b-4f22-bad1-4468d5b80eb5" />

    
    - 위의 전체 구조에서 알 수 있듯이 직렬로 (C→S) 연결되어있기 때문에 위의 Channel Attention Module에서 나온 feature map에 대해서 Average Pooling과 Max Pooling을 적용한다.
    - 마찬가지로 각 Pooling을 병렬로 처리한다.
    - 두 Pooling을 직렬 연산(WXHX1) → 7X7 Conv → Sigmoid(**WXHX1**) ⇒ Attention Score
    
    $$
    M_s(F)= \sigma(f^{7X7}([AvgPool(F);MaxPool(F)]))\\
    =\sigma(f^{7X7}([F^s_{avg};F^s_{max}]))
    $$
    

## 3. Evaluation

<img width="817" height="604" alt="image" src="https://github.com/user-attachments/assets/a71986f6-3197-470e-98a0-98adcbcb318d" />


- 가벼운 모델부터 무거운 모델까지 같은 모델에 SE block을 적용한 것과 parameter수는 동일하지만 FLOPs수가 감소하고 error는 큰 폭으로 감소했다.

<img width="1082" height="374" alt="image" src="https://github.com/user-attachments/assets/f74e1a02-dd65-48cb-9413-cbd81ec6ed58" />


- Train과 Validation 과정에서도 기본 → SE 적용 → CBAM 적용 순으로 error가 감소했다.

