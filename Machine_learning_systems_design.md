<Strong><u>Machine learning systems design</u></strong>

작업에 우선순위를 두는 것은 가장 중요한 skill이다.

**Prioritizing what to work on - spam classification example**

$y = 1$ -> Spam (misspelled word)

$y = 0$ -> Not spam  (real content)

우선, 스팸인 것과 아닌 것을 구분하는 100 words를 고른다. ex) buy, discount, deal, now$\cdots$

크기가 100인 feature vector $x$에 encode하는데 실제 이메일 내용에서 선택한 100단어 중 어느 하나가 등장하면 그에 해당하는 matrix element($x_j$)에 1, 아니면 0으로 값을 매긴다.

스팸메일로 일상 생활에서 자주 보는 단어를 manually 선택하는 것이 아니라, training set에서 스팸메일을 구분할 수 있는 가장 자주 나타나는 단어들을 feature로 골라야 한다.

**What's the best use of your time to improve system accuracy?**

1. 많은 데이터를 수집한다. 

   ex) Honey Pot Project : 가짜 이메일 주소를 만들어 스팸메일을 모아 데이터를 수집하는 프로젝트

   데이터가 많은 것이 거의 도움이 되지만 항상 도움이 되지는 않는다.

2. 이메일 라우팅 정보를 이용한 feature.

   스팸메일 발송자는 이메일의 출처를 모호하게 표현하고, fake email header을 사용한다.

3. 메시지의 내용을 분석하여 feature들을 정교화.

   ex) discount와 discounts, dealer와 Dealer

4. misspelling을 감지할 정교한 알고리즘을 사용한다.

   스팸메일 발송자는 고의적으로 오타를 넣어 스팸메일을 구분하기 어렵게 한다.

5. 위의 4가지 방법을 randomly하게 정하지 말고 정확성을 높이기 위해 시도할 수 있는 여러 가지 방법을 list up한다.

**Error analysis**

Error analysis는 성능을 높이기 위해 필요한 작업을 선택하는데에 도움을 주는 과정이다.

machine learning system을 구축하려면 빠르게 실행할 수 있는 간단한 알고리즘을 구축하는 것부터 시작하는 것이 좋다. 정교한 시스템은 아니지만 간단한 알고리즘부터 구현한 다음 cross vaildation data로 테스크해보는 것이 좋다. 이 과정을 마치면 learning curves를 그려보고 데이터와 feature 등이 알고르즘을 최적화하는데에 도움이 되는지 안되는지 확인이 가능하다. 이 후에 어떤 과정을 진행할 지에 대한 guideline을 알 수 있게 된다. 

Error analysis를 통해 misclassify하는 training data에 대한 patterns알 수 있다. 이 pattern로 misclassify하는 data를 카테고리화하여 분류할 수 있고 오류가 가장 많이 발생하는 pattern을 우선적으로 해결하는 것이 효율적이다.

알고리즘이 어느정도의 성능을 가지고 있는지에 대한 수치가 나타나는 numerical evaluation을 사용하는 것도 좋은 방법이다. Spam classifying에서 형태가 약간 다르지만 같은 의미를 가진 언어를 같은 단어로 인식하는지 알고 싶다면 stemming software를 사용하여 형태소를 분석하는 방법이 있다.  이 방법으로 각 단어에 대한 error값을 알 수 있게 된다. error를 줄이기 위해 algorithm을 선택하는 방법은 약간의 변화를 주고 이 변화에 대한 error값을 비교하여 어느 algorithm이 효율적인지 알아낼 수 있다. 



**Error metrics for skewed analysis**

ex) cancer classification

logistic regression으로 $y=1$ : Cancer, $y=0$ : Otherwise

예를 들어 95%의 정상인과 5%의 암환자가 있다고 가정하고, 항상 정상인이라고 예측하면 accuracy는 당연히 높지만 암환자를 예측하기 위한 알고리즘이 전혀 작동하지 않는 것을 것이다. test set에서 테스트한 결과 1%의 error만 발생하였지만 실제 암 환자의 수는 0.5%밖에 되지 않는다. 이러한 경우 실제 암 환자를 100% 분류해내지 못하는 문제가 발생할 수 있다.

이처럼 skewed한 class가 있을 때는 error analysis가 효과적이지 못하므로 precision and recall을 사용해야한다.

**Precision and Recall**

|                   | Actial class 1 | Actial class 0 |
| ----------------- | -------------- | -------------- |
| Predicted class 1 | True Positive  | False Positive |
| Predicted class 0 | False Negative | True Negative  |

1. Precision : 얼마나 자주 false alarm을 야기하는가?

   ex) 우리가 암환자라고 예측한 환자 중에 실제 암환자의 비율은 얼마인가?

   $\frac{\text{True Positives}}{\text{Predicted Positive}} = \frac{\text{True Positives}}{\text{True Positive}+\text{False Positive}}$

   1에 가까울수록 logisteic regression이 잘 되었다고 판단할 수 있다.

2. Recall : 알고리즘이 얼마나 잘 작동하는가?

   ex) training set에 실제 암환자 중에서 이들을 정확히 감지한 비율은 얼만인가?

    $\frac{\text{True Positives}}{\text{Actual Positive}} = \frac{\text{True Positives}}{\text{True Positive}+\text{False Negative}}$

   1에 가까울수록 logisteic regression이 잘 되었다고 판단할 수 있다.



**Trading off precision and recall**

암 환자인지 아닌지를 판명하는 임계값을 높이냐 낮추냐에 따라 precision과 recall의 균형이 달라진다.

1. 임계값을 높이면?

   암 환자가 확실한 경우에만 환자에게 알리기 위해 임계값을 0.5에서 0.8로 올리게 되면 false alarm이 줄어듦으로 precision은 높아지게 되고, 임계값이 0.8보다 작은 암환자(False Negative)에 대해서 분류할 수 없으므로 recall은  낮아진다. 

2. 임계값을 낮추면?

   암 환자를 놓치는 것을 방지하기 위해 임계앖을 0.5에서 0.3으로 낮춘다면 모든 암환자를 예측할 수 있기 때문에 알고리즘이 잘 작동하므로 Recall은 높아지지만, 암 환자가 아님에도 약간의 의심스러움으로 인해 암환자로 예측하는 경우 False alarm이 발생될 수 있으므로 Precision은 낮아진다. 

Precision과 Recall을 그래프로 그렸을 때 Precision의 값이 큰 것은 임계값이 큰 값이고, Precision의 값이 큼에 따라 상대적으로 Recall의 값은 작다. Recall의 값이 큰 것은 임계값이 작은 값이고, Recall의 값이 큼에 따라 상대적으로 Precision의 값은 작다. 

여기서, 임계값을 어떤 방식으로 정하는지, 여러가지 알고리즘으로 실험해 보았을 때 precision과 recall이 어떠한 값을 갖는 알고리즘을 선택해야하는지 의문이 생긴다.

ex) 

|             | Precision | Recall |
| ----------- | --------- | ------ |
| Algorithm 1 | 0.5       | 0.4    |
| Algorithm 2 | 0.7       | 0.1    |
| Algorithm 3 | 0.02      | 1.0    |

1. Average : $\frac{P+R}{2}$ 

   이 방식은 Algorithm 3과 같이 항상 $y=1$, 또는 $y = 0$으로 예측했을 때 한쪽으로 치우친 값을 갖는 알고리즘의 Average값이 가장 크므로 좋지 않은 방법임을 알 수 있다.

2. $F_1$score : $2\frac{PR}{P+R}$

   이 방식은 Precision과 Recall값 중 무엇이든 더 낮은 값을 갖는 것에 가중치를 부여하는 방법이다.

   따라서, $F_1$score가 크려면 Precision과 Recall 모두 값이 어느정도 커야한다. 일반적으로 $F_1$score을 single number evaluation metric으로 많이 사용된다. $F_1$score가 가장 높은 Alogorithm 1이 가장 적합한 알고리즘이다. 

임계값을 자동적으로 설정되게 하기 위해서는 다양한 범위의 임계값으로 cross validation set를 evaluate한 다음 가장 높은 $F$값을 갖는 임계값을 선택하는 과정이 필요하다. 



**Data for machine learning**

**Designing a high accuraccy learning system**

Percepton,Winnow, Memory based, Naive Bayes 알고리즘을 사용하여 여러 데이터 셋 크기로 실험을 해본 결과, 알고리즘은 매우 유사한 성능을 나타내지만 training set의 크기가 증가함에 따라 정확도도 증가하는 것을 알 수 있었다. 

하지만 조건은 feature x가 y를 정확하게 예측하기에 충분한 information이 있다고 가정해야 한다. 이 조건이 만족할 때에만 데이터가 더 많을 수록 알고리즘의 성능에 도움이 된다.

logistic regression과 같이 많은 파라미터가 있거나, linear regression과 같이 feature가 많거나, NN에 hidden layer가 많은 learning algorithm을 사용한다고 가정했을 때, 파라미터로 복잡한 function를 충분히 학습할 수 있기 때문에 bias 낮다. 

작은 크기의 training set을 사용하면 당연히 training error도 작을 것이다. 파라미터에 영향을 미치지 않을 정도로, 즉 overfit되지 않을 정도의 더 큰 training set을 사용한다면 training error가 test error에 가까워지며 test error 또한 작아지게 된다.

low bias한 알고리즘을 위해서는 complex algorithm을 사용하고, low varience한 알고리즘을 위해서는 큰 training set을 사용해야 한다.

bias도 낮으면서 variance도 낮은 안정적인 모델을 만드는 것이 중요하다.