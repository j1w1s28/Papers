# What Does BERT Look At? An Analysis of BERT's Attention

태그: NLP, XAI
상태: In progress - code

---

# Abstract

- attention mechanisim을 분석하는 방법에 대해서 제시
- BERT의 Attention heads 들은 구분자 토큰, 특정 Positional Offset, 또는 전체 문장에 대해서 집중하는 현상이 나타남(같은 layer에 있는 heade 들끼리는 집중하는 부분이 비슷)
- 본 논문이 위 현상에서 나아가 특정 Attention heads들이 문법(syntax), 그리고 상호참조(Coreference)의 언어적 개념을 표현하는 역할을 함
    - 예시)
        - 동사의 대상인 목적어에 집중, 관형사와 명사를 연결, 전치사의 대상을 연결
- Attention 기반으로 분류 문제를 분석을 제안

# 1. Introduction

- Attention Weight는 현재 단어에서 다음 표현을 게산할 때 특정단어가 에대한 가중치를 나타냄 → 자연스럽게 의미론적으로 Attention Weight 자체는 `Interpretable`함
- 본 논문은 BERT의 144개의 Attention Heads들을 분석
- 먼저 BERT의 Attentnion head의 행동을 분석 (Attention Head들의 행동에서 공통적인 패턴 발견)
    
    → 구분자 토큰, 특정 Positional Offset, 또는 전체 문장에 대해서 집중하는 패턴
    
- 신기하게도 BERT의 Attention 중 상당부분이 `구분자(SEP)`에게 집중했음 → 해당 토큰은 연산에 큰 기여를 하지 않는다고 알려짐
- Attention head를 하나의 Model 로 보면서 Attentnion mechanisim의 언어적 현상에 대해서 분석함(input : word, output : attention weight이 제일 높은 단어)
    - 이를 통해 문법상의 관계를 잘 분류하는 능력에대해서 평가함
    - 단일 attention head가 모든 문법 관계를 잘 표현하지는 않았지만 특정 head들은 특정 관계에 대해서 매우 잘 표현
        - 예시) 동사의 대상인 목적어에 집중, 관형와 명사를 연결, 전치사의 대상, 소유 대명사의 대상을 연결하는 문제에서 75%이상의 정확도를 보임
- 문법이나 상호참조관계에대해서 지도학습을 하지도 않았는데 이런 언어적 특징을 학습하는 것은 매우 놀라움
- 나아가서 Attention map을 input으로 받고 classifier의 분석하는 방법을 제시

# 2. Background : Transformers and BERT

- Transformer는 multiple layer로 구성 → 각  layer는 multiple attention heads를 가지고 있음
    - Attention Head는 Sequence를 input으로 받음 (n개 token일 때 , head또한 n개)
    - 각 head는 Key, Query, Value로 각 `선형변환`을 통해 Vector 로 표시
    - Head는 attention weight $`a_{ij}`$를 query 와 key의 `softmax-normalized dot product`(내적 연산 후 이를 softmax) 를 통해 계산
    - 최종 output $`o_i`$은 attention weight를 통해 value에 대해서 weighted sum을 한 값을 출력
    
    ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled.png)
    
- 위와 같은 attention 작동방식 때문에 attention weight는 다음 표현에서 각 token들이 얼마나 중요한 역할을 하는지 보여주는 값으로 보임
- BERT는 33억개의 영어 token을 가지고 `Masked language modeling` 과 `next sentence predication` 두개의 task를 통해 사전학습한 모델임
- 본 논문은 12 layer, 각 layer 12 attetnion heads로 구성된 BERT base 모델 사용

# 3. Surface Level Patterns in Attention

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%201.png)

## Setup

- 1000 개의 랜덤한 Wikipedia segment을 기반으로 attention map을 출력함
- BERT pre-training과 동일한 set up(단지 mask 처리를 안함)
- input 포맷은 다음과 같음  `[CLS]<문단1>[SEP]<문단2>[SEP]`

## 3.1 Relative Position

- 먼저 attention이 현재 단어 , 그전 단어, 이후 단어에 대해서 얼마나 집중하는 계산함
- 대부분의 head들은 현재단어에 집중하지 않았음
- 하지만 몇몇 head들은 이전 단어 그리고 이후 단어에 매우 높은 가중치를 부여했음 대부분 초기 layer
    - 2,4,7,8 layer에서의 특정 attention head 들은 평균적으로 0.5 이상의 weight를 이전 token에 부여
    - layer 1,2,2,3,6의 특정 head들은 다음 token에 0.5 이상의 attention을 부여

## 3.2 Attending to Seperator Tokens

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%202.png)

- 상당한 수 의 Attention은 특정 token을 참조
    
    ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%203.png)
    
    - 예시로 layer6~10 의 절반이상의  head들이 [sep] 에 참조함
    - 이를 해석해보면 대부분의 segment는 128개의 토큰으로 이루어져있고 SEP 토큰은 segment에서 2번 등장 하므로 평균적으로 약 1/64의 attention을 받는다.
    - SEP와 CLS 토큰은 나오는게 무조건 보장되어있으며 학습시에 masking 처리되지도 않음
    - `.` 이나 `,` 는 the를 제외하면 가장 많이 등장하는 token임
    - `SEP`, `CLS`, `.` 이나 `,`  토큰에 대해서 모델이 집중하는 이유는 자주 등장해서 이지 않을까 생각
    - 위와 같은 특징은 uncased BERT(input 텍스트를 소문자로 만들고, accent marker를 제거해서 input하는 모델)에서도 나타나는걸로 보아 학습의 결과라기 보단 `구조적인 문제에서 야기한 것으로 보여짐`
    - 이런 현상의 가설 중 하나는 SEP 토큰이 각 문장을 분리하므로, 특정 head가 SEP 토큰을 참조하고 나머지 다른 head들이 문장 단위의 정보에 참조할 수 있도록 한다는 점이다.
        - 그러나 본 논문에서는 이런 가설을 입증하지 못했음
        - 만약 해당 가설이 맞다면 SEP 에 참조하는 head들은 문장 단위 정보를 생성하고자 SEP 토큰에서 다시 문장 전체에 대해서 참조해야 함
        - `하지만` 실제로는 대부분(90%) 이상의 SEP 토큰을 참조하는 head은 다시 같은 SEP 토큰을 참조하거나 다른 SEP토큰을 참조
    - 추가로 질적연구에서는 구체적인 기능을 담당하고 있는 head가 해당 기능이 사용되지 않을 때는 SEP토큰을 참조하고 있는 것으로 보임
        - head 8-10에서 동사의 목적어는 그들의 동사를 참조하게 되는데, 이 때 명사가 아닌 토큰들은 대개 SEP를 attend하고 있음
            
            ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%204.png)
            
        - 따라서, 해당 attention head의 기능으로 적용되지 않는 토큰들은 "`no-operation`"과 같은 느낌으로 사용될 때 sepcial token을 attend하는 것으로 추측
        - 해당 가설을 더 조사하고자 본 논문 `IG`를 사용해서 feature Importance를 측정
            - BERT Masked language modeling task에서의 각 attention weight에 대한 손실함수의 gradient를 측정
            - 해당 gradient 값은 토큰에 대한 attention을 얼마나 많이 변화시켜야 버트의 결과가 바뀌게 되는지를 측정할 수 있음 (아래그림 참조)
            - 그림에서 볼 수 있듯이 SEP에 대한 attention은 매우 높지만 실제로 SEP의 attention에 대한 gradient는 매우 낮음
            - 즉 SEP를 많이 attend하든 또는 적게하든 BERT의 output에는 `큰 영향을 주지 않는다`는 점을 시사함 →`no-operation`"과 같은 느낌으로 사용된다는 가설을 뒷받침
                
                ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%205.png)
                
    

## 3.3  Focused vs Broad Attention

- Attention Heads들이 특정 단어에 집중하는지 여러 단어를 두루두루 집중하는지 측정함
- 이를 위해 각 attention head의 분포도에서 평균 entropy를 계산 (그 값이 클 수록 데이터가 혼재되어있다는  즉 두루두루 집중한다느 뜻?)
- 엔트로피 공식
    
    ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%206.png)
    
- 초기 layer에서의 몇몇 attentino heads 들은 매우 넓은 attention을 보임 → 높은 entropy 값
    - 이때는 일반적으로 많아야 10%의 attentnion weight값을 단일 word에 부여
    - 해당 head의 output은 문장에서의 vector 집합의 표현을 보여주는 것 같았음
    
    ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%207.png)
    
- CLS 토큰에서 부터 모든 attention heads들의 entropy를 게산
    - 이는 위 그림과 굉장히 비슷한결과를 보임
    - 마지막 layer의 CLS 토큰에대한 attention들의 entropy 값은 3.89로 매우 넓은 attention을 나타
    - 이러한 결과는 CLS 토큰에 대한 representation이 사전학습시 `NextSentencePredication task의 input으로 사용된다는 점`에서 이해할 수 있는데, 왜냐하면 CLS 토큰의 representation이 전반적으로 attend 하면서 마지막 layer의 전체적인 input의 representation을 집계

# 4. Probing Individual Attention Head

- 각 Attention head를 분석해서 이들이 언어의 어떤 측면을 학습했는지 알아봄
- dependency parsing(단어간의 의존 관계를 파악하는 task ) 과 같은 task data를 활용해 evaluate 진행

## 4.1 Method

- word level에서 attention heads을 평가하려했지만 BERT는 Byte-pair tokenization을 활용하기 때문에 한단어가 몇몇 token으로 분리되는 형식
- 그렇기 때문에 token-token attention 형식을 → `word-word attention map`으로 `변환`
- 분할된 word의 attention 의 경우 `attention weight의 합`을 이용
- 분할된 word에서의 → 단일 토큰 에 대한 attention 값은 각 `attention weight의 평균`을 활용
- 이렇가하면 각 단어의 attention weight 값의 합이 1로 보존해줌
- 해당 attention head와 word 에 대한 가장 높은 attention weight을 가지는 `word`를 모델의 예측값으로 사용함(SEP 와 CLS는 무시했음 → 바꾸더라도 대부분의 head의 정확도를 크게 바꾸지 않았음)

## 4.2 Dependency Syntax

### Setup

- Stanford Dependencies로 annotate된 Penn Tree bank 중 하나인 월 스트리트 저널의 일부분을 버트에 적용하여 `attention map을 추출`
- 각각의 attention head의 예측의 `두 방향을 모두 평가`(head word의 dependent에 대한 attention과 dependent의 head word에 대한 attention)
    - 몇몇 dependency 관계를 예측하는 것은 매우 쉬움 ex) 명사에 대한 관형사는 대체로 바로 앞단어
    - 그렇기 때문에 Simple-Fixed offset baseline에서부터의 prediction 결과를 보임으로써 비교를 진행 ex) a fixed offset of -2 는 dependent의 왼쪽 2칸에 있는 어떤 단어는 항상 head word로 간주한다는 의미 → `뭔말이래?`

### Results

- Table1에서 볼 수 있듯이 단일 attention head가 문법 “Overall” 에대해서 잘하지는 못했음 (`UAS 34.5`) 으로 기존 baseline 26.3 UAS 보다 그다지 높지 않음
- UAS, Dependency Parser
    
     unlabeled attachment score (UAS) : arc(dependency) 방만 일치하는지 확인한다. label은 별도로 확인하지 않는다.
    
    [https://misconstructed.tistory.com/33](https://misconstructed.tistory.com/33)
    
- 반면에, 특정 head는 특정 dependency relation을 다루고 있고, 때때로 매우 높은 정확도와 baseline과 비교해 매우 높은 점수를 얻음
    - 각각의 dependent는 반드시 하나의 head를 가지는 반면에, 각 head는 여러개의 dependent를 가질 수 있으므로, `pobj`를 제외하고는 테이블 1에 있는 모든 관계가 dependent가 head word에 attentd 하고 있었음
    - 아마 각 dependent는 하나의 head word를 가지고 있지만  각 head word들은 여러개 dependent를 가질 수 있기 때문일 것이다.
- 또 기존의 annotation과는 다른 결과가 나오기도 함 ex) head 7 - 6는 `'s`를 `poss` 관계의 dependent로 보았지만, 기존 표준 annotation은`'s`의 보어를 `dependent`로 정의
- 이러한 차이는 BERT가 단순히 인간이 정의한 틀 안에서가 아닌, self-supervised 학습에 의해 얻은 결과를 가지고 학습한다는 것을 보여줌
- 그림 5는 몇가지 attention behavior의 예시를 보여줌
- 모델이 학습한 attention weight와 인간이 정의한 통사 관계의 유사성은 놀랍지만, attention head마다 특히 좋은 성능을 내는 관계가 따로 있다. 간단한 베이스라인 코드보다 버트가 약간만 성능이 좋아지므로 개별적인 어텐션 헤드가 전체적인 dependency structure를 파악한다고 말할 수는 없음
    - attention이 잘 파악할 수 있는 관계가 다른 언어에서도 동일한지 아닌지를 분석해보면 우리의 분석을 확장할 수 있는 재미있는 추후 연구가 될 것
    
    ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%208.png)
    

## 4.3 Coreference Resolution

- semantic의 측면에서 coreference resolution(이하 co-res.)을 다뤄보고자함
- coreference links는 대개는 syntactic dependencies보다 훨씬 길며 최신 기술로도 parsing에 비해 좋지 않은 성능을 냄

### Setup

- CoNLL-2012 dataset을 사용하여 co-res.에 대한 attention head를 평가
- antecedent selection accuracy 라는 평가방식으로 계산
    - coreference mention 중의 주요 단어가 이러한 mention들의 antecedent(선행자)중 하나의 head에 얼마나 많은 비중으로 attend 하는지 계산
    - Coreference Mention
        
        아래의 글에서 파란 글씨로 보이는 것이 coref. mention들이다.
        
        ![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%209.png)
        
        [https://happyzipsa.tistory.com/13](https://happyzipsa.tistory.com/13)
        
        Antecedent
        
        - 위 글의 첫줄에서 `his`는 referent, `Obama`는 antecedent, 선행자이다.
    - 우리는 antecedent를 뽑는데에 3가지 베이스라인 방법과 비교
        - 가장 가까운 다른 mention 고르기
        - 똑같은 same head word를 가지고 있는 가장 가까운 mention 고르기
        - Rule-based에 의해 고르기. 다음의 우선순의 조건을 먼저 만족하는 가장 가까운 mention 고르기 (1) full string match (2) head word match (3) number/gender/person match (4) all other mention

### Results

- BERT의 attention heads들 중 하나는 Coreference Resolution에서 좋은 성능을 보임
- 베이스라인 보다 10 이상의 정확도 점수를 높였고 rule-based systemp에 가까운 성능을 보임
- 특히 명사같은 mention에서 좋은 성능을 냈음 → 그림 5의 우측 하단에서 보는 바와 같이 동의어들 사이의 미세한 관계를 잡을 수 있었기 때문인 것 같음

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%2010.png)

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%2011.png)

# 5. Probing Attention Head Combinations

- 개별의 attentnion head 들이 syntax의 특정 부분을 제각각 담당하고 있기 때문에 전체적인 모델의 지식은 여러 어텐션 헤드에 퍼져있음
- 통합적인 능력을 평가하기 위해서 우리는 새로운 attention-based의 그룹 probing classifiers를 제안
    - dependency parsing을 담당하며, 분류를 위해 버트의 어텐션 output은 고정
    - 몇개의 파라미터만 조금 훈련시키고 역전파 과정은 `BERT`를 거치지 않음
    - probing classifiers는 기본적으로 graph-based dependency parser
    - input : word
    - output : input word와 같은 head에 존재하는 각각의 다른 단어들이 동일한 의미를 지니는지 나타내는 확률 분포를 반환

## Attention- Only Probe

- attention weights의 간단한 선형 결합을 학습
- *p*(*i*∣*j*)는 단어 i가 단어 j의 syntactic head일 확률
- $a^k_{i,j}$는 단어 i에서 단어 j로의 head k로 부터 생성되는 attention weigh
- $a^k_{j,i}$어텐션은 양방향이기 때문에, candidate head에서 dependent head 뿐만 아니라 반대로도 고려
- n은 attention head의 개수
- w , u 는 지도학습으로 학습됨

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%2012.png)

## Attention and Words Probe

- 각 Heads들이 특정 syntatic relation에 특화되어있기 때문에 probing classifiers는 input word로부터 정보를 얻어 활용하면 좋을 것이라 생각
- GloVe 기반 Embedding을 어텐션 head의 가중치로 사용
- 이는 직관적으로 dependent 'the'와 candidate 'cat'에 대해서 probing classifier는 대부분의 weight을 determiner relation를 관장하는 head 8-11에 할당해야하는 점을 나타냄

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%2013.png)

- v는 GloVe 임베딩을 ⊕는 concat을 의미한다. GloVe 임베딩은 학습 시 고정되기 때문에 W와 U만 학습되며 내적연산은 특정 head에 적용되는 word-sensitive weight를 만듬

## Results

- Stanford dependencies가 annotating한 Penn Treebank dev set에 기반한 방법론으로 평가
- 다음과 같은 3개의 베이스라인과 비교
    - head가 항상 dependent의 오른쪽에 있는 right-branching baseline
    - dependent와 candidate의 GloVe 임베딩을 입력으로 받는 simpe one-hidden-layer network
    - 우리의 attention-and-words probe. 이 때 bert의 word/positional embedding을 사용하지만, 이들은 모두 `랜덤 초기화`. 이러한 baseline들은 다른 probing tasks에서 좋은 성능을 보여준다.
- Attn + GloVe가 매우 성능이 좋았음 →BERT attention map이 영어 syntax의 표현을 엄청나게 잘한다는 것을 시사
- 기존의 Syntax - aware - model은 architecture design을 통해 개발되어 오거나 human - curated treebanks(인간이 만든 지침 기반으로 만든 의사 결정 트리)를 통한 지도학습으로 개발
- 그러나 풍부한 `pre-training task`를 통해 `language model의 언어의 계층적구조를 간접적으로 학습` 할 수 있다는 점을 나타남

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%2014.png)

# 6. Clustering Attention Heads

- 동일한 layer에서의 Attention Heads 의 행동은 유사한지 상이한지 확인
- 모든 Attention Heads들의 거리를 비교하면서 확인함
- 거리 계산에는 아래 식을 사용(`Jensen-Shannon Divergence` 를 사용)

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%2015.png)

- 그림 6엥서 볼 수 있듯이 몇몇 head들의 cluster들은 유사해보이고, 행동또한 유사했음
- 같은 layer에 있는 header들은 비슷한 attention 분포를 보였

![Untitled](What%20Does%20BERT%20Look%20At%20An%20Analysis%20of%20BERT's%20Atten%209b7dc193d39a45cb88a6670a0552942e/Untitled%2016.png)

# 8. Conclusion

- 모델의 Attention 매커니즘을 분석하는 방법을 제안
- 언어적  지식이 attention map에서 발견됨
- Attention Map을 분석하는 방법들은 다른 모델 분석에 충분히 보완할 수 있을 것임

---

참조

[https://velog.io/@sangmandu/What-Does-BERT-Look-AtAn-Analysis-of-BERTs-Attention](https://velog.io/@sangmandu/What-Does-BERT-Look-AtAn-Analysis-of-BERTs-Attention)