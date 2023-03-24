# ViT  
## 1. 모델 설계  
계층 구조를 아래와 같이 설계하였습니다.  
```
VisionTransformer
    - PatchEmbeddingBlock
    - EncoderBlock
        -- MultiHeadSelfAttentionBlock
        -- MlpBlock
    - Classifier
```
  
## 2. 구현 포인트  
* N개의 qkv를 생성하는 linear layer는 N의 개수와 무관하게 qkv당 각 1개씩만 존재합니다.  
* Encoder block의 skip conn에는 dropout이 존재합니다.  
* attention의 activation function은 MLP에 단 한개밖에 없습니다.  
* classifier의 feature는 패치 average가 아니라 embed_dim의 average입니다.  
* eniops를 적극적으로 사용하였습니다.  
  
## 3. 주의 사항  
* 포지셔널 인코딩을 learnable random params로 변경하였습니다.  
* 클래스 토큰을 learnalbe random params로 변경하였습니다.  
* 클래스 토큰을 classification의 피처로 이용하지 않고,  
 embed_dim의 avg를 cls의 피처로 이용했습니다.
