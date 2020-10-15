**Advice for applying Machine Learning** 

<u>Deciding what to try next</u>

많은 algorithm중에 어떤 algorithm을 사용하는 것이 제일 효율적일지 결정하는 능력이 중요하다.

예를 들어, housing prices를 예측하는 문제에서 regularized linear regression algorithm을 이용하여 구한 $J(\theta)$가 새로운 test set에 적용하였을 때 큰 error를 발생한다고 하자. 이 때, 우리가 취할 수 있는 조치는 총 5가지가 있다.

1. Get more training set
2. Try a smaller set a features
3. Try getting additional features
4. Building your own, new, better features
5. Try decreasing or increasing $\lambda$

 많은 사람들은 이 5가지 방법 중 하나를 직감적으로 고르기 때문에 시간을 투자한 것에 비해 효율이 없다.  이 5가지 항목 중 시간을 낭비할 절반의 선택지를 배제할 수 있는 방법은 **Machine learning diagnostics**이다.

Machine learning diagnostics는 algorithm이 잘 작동하는지 아닌지를 판단할 수 있는 통찰력에 대한 테스트와 성능을 향상시키는 guidance를 얻기 위한 테스트이다. 이것을 이해하는 데에 시간이 걸리기는 하지만 제대로 작동하지도 않을 방법에 투자하는 것보다 훨씬 효율적이다.



**Evaluating a hypothesis ** 

우선, data를 2개의 portion으로 나눈다. 1번째는 training set, 2번째는 test set이다. training set와 test set의 비율은 7:3이 가장 일반적이다. 데이터들이 정렬되어있다면 random하게 shuffled한 후에 training과 test set을 나누는 것이 좋다.

그 다음에 training set으로부터 $h_\theta(x)$를 도출하여 $J(\theta)$를 최소화하는 파라미터 $\theta$값을 구하여 test error를 계산한다. 

1. linear regression

   $J_\text{test}(\theta) = \frac{1}{2m_\text{test}}\Sigma_{i=1}^{m_\text{test}}(h_\theta(x_\text{test}^{(i)})-y_\text{test}^{(i)})^2$

2. logistic regression

   $J_\text{test}(\theta)= -\frac{1}{m_\text{test}}[\Sigma_{i=1}^{m_\text{test}}(y_\text{test}^{(i)}logh_\theta(x_\text{test}^{(i)})+(1-y_\text{test}^{(i)})logh_\theta(x_\text{test}^{(i)})]$를 

   misclassification을 이용하여 더 간단히 나타내면 $err(h_\theta(x),y) = \begin{cases} 1 \ \mbox{if $h_\theta(x) \geq 0.5$, $y=0$ } \mbox{or if $h_\theta(x) < 0.5$, $y=1$ }\\ 0 \ \mbox{otherwise} \end{cases}$이다.

   $h_\theta(x)$가 0.5보다 크면 y=1이고 $h_\theta(x)$가 0.5보다 작으면 y =0이기 때문에 이와 반대되는 값을 가지면 error 값에 1을 더해주고 나머지 참값은 0으로 error값에 영향을 미치지 않도록 한다.

   따라서, $J_\text{test}(\theta)= -\frac{1}{m_\text{test}}err(h_\theta(x_\text{test}^{(i)}),y_\text{test}^{(i)})$이다. 



**Model selection and training validation test sets**

Regularization을 위해 알맞은 파라미터를 고르는 것을 Model selection problems라고 한다.

$d$ : hypothesis를 generalize할 가장 적합한 모델의 degree

ex) 알맞은 polynomial을 선택하는 방법

​	$h_\theta(x) = \theta_0+\theta_1x$

​	$h_\theta(x) = \theta_0+\theta_1x+\theta_2x^2$

​		$\vdots$

​	$h_\theta(x)=\theta_0+\theta_1x+\cdots+\theta_{10}x^{10}$ 

각각의 $h_\theta(x)$에서 $J(\theta)$를 최소화하는 파라미터 $\theta$를 구하고 $\theta^{(i)}$라고 표현한다. 마찬가지로  $\theta^{(i)}$로 $J_\text{test}(\theta^i)$를 구한다. 그러면 각각의 degree에서 $J_\text{test}(\theta^i)$를 구하고 그 중에서 가장 작은 hypothesis를 알 수 있다.

하지만 이러한 방식은 $J_\text{test}(\theta)$가 test set과 차이가 적은 값으로 결정된 degree이기 때문에 새로운 data가 있다면 다른 degree가 더 적합할 수도 있다.  따라서, 이 방법으로 구한 $J_\text{test}(\theta)$는  generalization error로 볼 수 없기 때문에 좋은 방법이 아니다. 

이 점을 보완하기 위해 **cross validation**을 적용한다.

가지고 있는 data에서 60%를 training set, 20%를 cross validation, 20%를 test set으로 한다.

linear regression에서 Training error, Cross Validation, Test error를 구하면 다음과 같다.

​	Training error : $J_\text{train}(\theta) = \frac{1}{2m}\Sigma_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$

​	Cross Validation error : $J_\text{cv}(\theta) = \frac{1}{2m_\text{cv}}\Sigma_{i=1}^{m_\text{cv}}(h_\theta(x_\text{cv}^{(i)})-y_\text{cv}^{(i)})^2$

​	Test error : $J_\text{test}(\theta) = \frac{1}{2m_\text{test}}\Sigma_{i=1}^{m_\text{test}}(h_\theta(x_\text{test}^{(i)})-y_\text{test}^{(i)})^2$

$J(\theta)$를 최소화할 $\theta$를 구하여 Cross Validation set으로 $J_\text{cv}(\theta)$를 구한다. 그 후, 가장 작은 cross validation error를 갖는 hypothesis를 선택한다. 

이러한 방식은 test set에 정확히 딱 맞는 hypothesis를 선택하는 것이 아니라, cross validation을 이용하여  Cross Validation error가 가장 작은 hypothesis를 선택하도록 한다. 따라서,  test set error 즉, generalization error를 구할 수 있다.



**Diagnosis - bias vs. variance**

1. $J_\text{train}(\theta)$

   degree of polynomial d가 커질수록 overfit되므로  $J_\text{train}(\theta)$가 작아짐.

   degree if polynomial d가 작아질수록 underfit되므로 $J_\text{train}(\theta)$가 커짐.

2. $J_\text{cv}(\theta)$ or $J_\text{test}(\theta)$

   degree of polynomial d가 1일 때는 underfit되어 $J_\text{cv}(\theta)$가 매우 커짐 <- uinderfit

   degree of polynomial d가 2일 때 $J_\text{cv}(\theta)$가 가장 작아짐

   degree of polynomial d가 3 이상일 때, $J_\text{cv}(\theta)$가 점점 커짐 <-overfit

1. **high bias(underfit)**

   d가 매우 작고, $J_\text{train}(\theta)$가 매우 크고, underfit되기 때문에 $J_\text{cv}(\theta)$도 매우 크다.

2. **high variance(overfit)**

   d가 매우 크고, $J_\text{train}(\theta)$가 매우 작고, overfit되기 떄문에 $J_\text{cv}(\theta)$도 매우 크다.



**Regularization and bias/variance**

Linear regression with regularization

$h_\theta(x) = \theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$

$J(\theta) = \frac{1}{2m}\Sigma_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2 +\frac{\lambda}{2m}\Sigma_{j=1}^m\theta_j^2$

1. $\lambda$ is large

   모든 파라미터 $\theta$가 penalized되어 0에 가까워지고 결과적으로 $h_\theta(x) = \theta_0$이 된다.

   이것은 underfitting 되어 high bias하다.

2. $\lambda$ is small

   regularization term이 0에 되어 정규화를 하지 않은 것과 같다.

   이것은 overfitting 되어 high variance하다.

ex) 알맞은 $\lambda$를 선택하는 방법

​	$model(1) = \lambda = 0$

​	model(2) = \lambda = 0.01$

​	model(3) = \lambda = 0.02$

​		$\vdots$

​	model(p) = \lambda = 10$

p번째 $\lambda$를 적용하여 $J(\theta)$를 최소화하는 $\theta^{(p)}$를 구하고 이 파라미터로 $J_\text{cv}\theta^{(p)}$를 구한다. error의 값이 가장 작은 것을 선택한다. 그 후 test set을 이용하여 test error를 구한다.

1. high bias

   $\lambda$ 가 매우 크고, underfitting되기 때문에 $J_\text{train}(\theta)$, $J_\text{cv}(\theta)$가 매우 크다. 

2. high variance

    $\lambda$가 매우 작고, overfitting되기 때문에 $J_\text{train}(\theta)$는 매우 작은 반면에 $J_\text{cv}(\theta)$는 매우 크다. 



**Learning curves**

training example수를 $m$으로 하고 이 $m$을 x축, error값을 y축으로 하는 그래프에 $J_\text{train}(\theta)$, $J_\text{cv}(\theta)$를 그려보며 training example수가 $J_\text{train}(\theta)$, $J_\text{cv}(\theta)$에 어떤 영향을 끼치는지 알 수 있다.

1. $J_\text{train}(\theta)$

   $m$이 작을수록 $J_\text{train}(\theta)$도 작아지고, $m$이 클수록 $J_\text{train}(\theta)$도 커진다. 

   training set이 작을수록 정확하게 training set에 맞는  $h(x)$를 구할 수 있기 때문이다. 

2. $J_\text{cv}(\theta)$

   $m$이 작을수록 $J_\text{cv}(\theta)$가 커지고, $m$이 클수록 $J_\text{cv}(\theta)$가 작아진다. 

   trining set이 작을 수록 $h(x)$를 generalize하기 어렵고 training set이 클수록 $h(x)$를 generalize하기 쉽기 때문이다. 

1. high bias

   $m$이 작을 수록 $J_\text{train}(\theta)$은 작지만 $m$이 클수록 trining set에 딱 맞는 hypothesis를 찾기 어렵기 때문에  $J_\text{train}(\theta)$는 커진다. 여기서 주목할 점은 $J_\text{train}(\theta)$의 기울기가 가파르게 증가한다는 점이다. $m$이 커지면 $J_\text{train}(\theta)$의 값은 상대적으로 크다.

    $m$이 작을수록 generalize하기 어렵기 때문에 $J_\text{cv}(\theta)$는 크다.  $m$이 커질수록 generalize는 쉽기 때문에 $J_\text{cv}(\theta)$는 작아진다.

   $J_\text{train}(\theta)$, $J_\text{cv}(\theta)$ 를 그려보면 $m$이 커질수록 $J_\text{train}(\theta)$와 $J_\text{cv}(\theta)$가 gap이 작아지고 상대적으로 둘 다 error가 큰 쪽으로 수렴되는 것을 알 수 있다.

   따라서, high bias의 경우 training set을 늘리더라도  $J_\text{train}(\theta)$, $J_\text{cv}(\theta)$가 줄어들지 않기 때문에 도움이 안된다는 것을 알 수 있다.

2. high variance

   $m$이 작을 수록 $J_\text{train}(\theta)$은 작지만 $m$이 클수록 trining set에 딱 맞는 hypothesis를 찾기 어렵기 때문에  $J_\text{train}(\theta)$는 커진다. 여기서 주목할 점은 $J_\text{train}(\theta)$가 완만하게 증가한다는 점이다. $m$이 커지더라도 $J_\text{train}(\theta)$의 값은 상대적으로 작다.

    $m$이 작을수록 generalize하기 어렵기 때문에 $J_\text{cv}(\theta)$는 크다.  $m$이 커질수록 generalize는 쉽기 떄문에 $J_\text{cv}(\theta)$는 작아진다.

   하지만 high bias와 다르게 $J_\text{train}(\theta)$, $J_\text{cv}(\theta)$사이의 gap이 크고 상대적으로 $J_\text{train}(\theta)$가 작은 값을 가지기 때문에 작은 error로 수렴하게 된다. 

   따라서, high variance의 경우 trining set을 늘리는 것이  $J_\text{train}(\theta)$, $J_\text{cv}(\theta)$가 줄어들기 때문에 도움이 된다는 것을 알 수 있다.