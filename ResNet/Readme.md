# ResNet 논문 리뷰

## “Deep Residual Learning for Image Recognition”

- 모델을 매우 깊게 만드는데 성공
- **skip-connection**을 적용하면 깊을수록 error가 감소함을 보임

---

## 0. Why skip-connection?

- Batch Normalization과 ReLU가 Vanishing Gradient 문제를 해결했지만, 모델이 깊을수록 training error가 커지는 문제 발생
- 이를 해결하고자 ‘깊은 것이 얕은 겻을 흉내낼 수 있지 않은가?’라는 아이디어로 시작
    
    → 이것이 곧 ‘**identity mapping**’: Input 그대로 Output 값이 되도록 하는 것. 
    
- 하지만, 오히려 training error가 커지는 것을 보아 모델이 identity mapping을 만들기 어려워한다는 것을 발견.
    
    ![image.png](attachment:3e2cea07-e8d3-4cc3-9d9d-654753a09cb9:image.png)
    
    → 그래서 등장한게 ‘**skip-connection(shortcut connection)**’인 것.
    
    → Vanishing Gradient를 해결하기 위해서 skip-connection을 하는 것이 아닌 skip-connection을 적용해보니 효과가 뛰어났던 것. 
    

## 1. Identity mapping

- 위 사진을 예를 들어, 20-layer짜리 모델을 56-layer짜리 모델로 흉내내기 위해서 20층을 제외한 나머지 36층은 모두 입력값 그대로 출력값으로 나오도록 설계하는 것이다.

## 2. skip-connection

![image.png](attachment:aabdda4d-637c-48a2-a1ad-9e3898e242ac:image.png)

- 입력값 $x$가 들어와서 $F(x)$가 나가는 것이 기본. 여기서, $x \ +\ F(x)$가 나가도록 layer를 연결하는 것이 skip-connection이다.
- 효과가 있는 이유
    - 만약 $x$로 $H(x) \ = x$(identitiy mapping)을 만든다고 해보자.
    - skip-connection이 없는 MLP라면 weight matrix $F(x)$는 ‘항등 행렬’이 되어야 할 것이다. → $H(x) = F(x)$
    - skip-connection이 있는 MLP라면 weight matrix $F(x)$는 ‘영 행렬’이면 된다. → $H(x)=F(x)+x$
    - 그런데, weight_init은 0과 근사하게 하기 때문에 skip-connection이 있는 MLP가 목표한 $H(x)$를 가장 효과적으로 만들 수 있다.
    - 또한 만약 역전파 과정에서 $F'(x)$=0이더라도 추가된 $x$도 미분되어 1이 항상 더해지므로 gradient가 0이 되는 것을 막아 VG를 방지할 수 있게된다.
- 이름의 유래: 입력값과 출력값 사이의 차이만을 학습한다는 것에서 ‘잔차 ⇒ residual’ 학습을 적용한 Network이므로 ResNet!

## 3. Model Structure

![image.png](attachment:9b066278-f900-4356-9b77-6e4dff3a3829:image.png)

- Conv 이후엔 ReLU는 작성 생략
- BN을 한다면) Input → Conv → BN → ReLU → Conv → BN → **+ Input** → ReLU 순으로 진행
- /2는 Size reduction을 위한 stride=2를 의미 (pooling 대신 사용)
- 점선: 1X1 Conv, stride=2 (projection connection)
- 실선: identity shortcut (***in_channel = out_channel이여야 가능***)
- 마지막엔 GAP를 적용

![image.png](attachment:bcb92d3a-2616-4e33-9fb2-6dcc335abc2b:image.png)

- 50층 이상의 ResNet부터는 **Bottleneck** 구조를 사용하여 params. FLOPs 감소
- 1X1으로 channel 수를 조절하면서 3X3로 공간 정보를 학습

![image.png](attachment:db66d649-a7cf-46bf-bdde-94acc147265e:image.png)

- 34-layer 모델까진 conv_x의 첫 번째 layer에서 stride=2를 적용하고 50-layer 모델 부턴 conv_x의 두 번째 layer에서 stride=2를 적용

---

- Example) 표 읽는 법
    
    만약 RGB Image를 50-layer에 입력한다면 Channel=3 (채널 수만 작성함)
    
    [RGB]:  3
    
    [conv1]: 3 → 64
    
    [conv2_x]: 64 → (64 → 64 → 256) 점선 → (256 → 64 → 64 → 256) 실선 →
    
    (256 → 64 → 64 → 256) 실선 
    
    [conv3_x]: (256 → 128 → 128 → 512) 점선 → 실선 X 3 … (반복)
    
    .
    
    .
    
    .
    
    [GAP] → 2048 → [softmax] → 1000 
    
    → 점선, 실선 헷갈리면 위의 개념 다시 상기해보기
    

---

## 4. evaluation

![image.png](attachment:88f3006e-63b5-47b3-b736-0bc395a5ca31:image.png)

- 깊을수록 std가 더 작다는 것을 알 수 있다. 더 안정적이다.
- plain한 모델들은 기존의 문제인 깊을수록 불안정하다. (std가 크다)

![image.png](attachment:fd1a60ed-5332-484f-a0b8-858d09d6a90c:image.png)

- 위 실험 결과도 마찬가지로 기존의 모델들은 깊을수록 error가 높지만 skip-connection을 적용한 모델들은 깊은수록 오히려 성능이 좋아진다.
- 좋아지지는 않더라도 비슷, 동일하기만 해도 상당한 발견이지만 오히려 성능 향상에 기여하는 skip-connection은 이후에 나온 수많은 모델들의 필수 사항이 되었다.

![image.png](attachment:dfe40590-0817-4cec-b81a-277c684d437d:image.png)

- 6개의 모델을 앙상블하여 이미지 분류에서 1등 달성.

## 5. Points of Confusion

### 1. 1개층만 건너뛰는 것은 왜 하지 않는가?

- 수용영역과 표현력을 확보하기 위해서 2층, 3층 단위를 채택했다.

### 2. 왜 2개층을 건너뛰는 구조와 3개층을 건너뛰는 구조를 섞어서 사용하지 않고 완전히 분리해서 사용하나?

- GPT:

**ResNet은 “스테이지별 동일 블록 반복”으로 단순함·안정성·효율을 얻는 설계에**요.

Basic(2개)과 Bottleneck(3개)은 **expansion·스케일링 철학이 달라** 한 모델 안에서 섞으면 규칙이 깨지고 이득도 크지 않아서, **아예 모델 패밀리 차원에서 분리** (18/34 vs 50/101/152)하는 게 표준입니다.