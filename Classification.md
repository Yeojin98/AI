<strong><u>Classification</u></strong>

예측하고자하는 y변수가 이산형 값(discrete value)을 가지는 경우

y는 0 또는 1의 값을 가지는 경우에는 binary class라고 하고 여러가지 y값을 가지는 경우는 multicalss라고 한다. 

0 : negative (absence of something), 1: positive (presence of something)



ex) binary class problem : Tumour size vs malognancy (0 or 1)

만약 linear regression을 사용한다면?

<u>문제점1</u>. 하지만 오른쪽으로 치우쳐진 data가 추가된다면?

-> training set에 맞춰  직선을 그어보며 얻은 $h_\theta(x)$함수에서 임계값을 0.5로 설정할 수 있다. 따라서, $h_\theta(x) \geq 0.5$이면 y = 1,  $h_\theta(x) < 0.5$이면 y = 0이라고 예측할 수 있다.

-> 임계값이 0.5보다 커지게 되어 예를 들어 0.6정도라고 가정하면 0.5와 0.6 사이에 있는 data에 대해서는 음성 판정을 하게 되고 이는 실제 데이터 값에 모순이 되는 결과를 나타낸다.

<u>문제점2.</u> 우리가 예측한 $h_\theta(x)$의 값이 1보다 크고, 0보다 작다면?

-> binary class는 0과 1이라는 정해진 결과값만이 가질 수 있는데 다른 값을 가지게 되면서 binary class의 의미가 모호해 질 수 있다.

=> 따라서, 위의 두 문제점으로 인해 예측하고자 하는 y변수가 discrete한 값을 가질 경우에는 logistic regression을 이용해야한다.



<strong><u>Hypothesis representation</u></strong>

classification에서 사용되는 $h_\theta(x)$의 표현

linear regression에서 사용했던 h를 이용한 $h_\theta(x) = g((\theta^Tx))$,

sigmiod fuction 또는 logistic function으로 불리는 $ g(z) = \frac{1}{1+e^{-z}}$ (z is a real number)

=>  두 식을 합한 형태인 $h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$로 표현한다.

sigmoid function은 z 값의 수평축을 기준으로 음의 무한대로 향하면 0에 수렴하고, 양의 무한대로 향하면 1에 수렴하여 0과 1사이에 값만을 가진다는 것을 알 수 있다. 따라서, 우리가 예측하고자 하는 값이 discrete한 값을 가질 경우에 적합하게 사용된다.

$h_\theta(x) = P(y=1|x;\theta)$로 표현하고 이것은 파라미터 $\theta$에 대하여 data x가 주어졌을 때 y가 1일 확률로 해석한다.

binary classification에서는 y가 0 또는 1의 값만 가질 수 있다는 것에 따라 $P(y=1|x;\theta)+P(y=0|x;\theta) = 1$인 것을 알 수 있다.



<strong><u>Decision boundary</u></strong>

sigmoid function 그래프를 살펴보면 $g(z) \geq 0.5$일 때 y = 1이라고 예측가능하고, $g(z)<0.5$일때 y =0이라고 예측할 수 있다. 

$g(z)\geq0.5$인 경우에서, z는 항상 양수 이다.  $z=\theta^Tx$이므로  $\theta^Tx \geq 0$이다. 따라서 다시 말하면 $\theta^Tx \geq 0$일 때, y = 1로 예측가능하다. 반대로 $g(z)<0.5$인 경우에는 $\theta^Tx \leq 0$ 일 때, y = 0으로 예측가능하다. 

$g(Z)=0.5$인 경우에는 



ex) $h_\theta(x) = g(\theta_0+\theta_1x_1+\theta_2x_2)$

파라미터 $\theta_0=-3, \theta_1=1, \theta_3=1$로 가정하자. 

그렇다면 $z=\theta^Tx = \begin{bmatrix} -3 & 1&1\end{bmatrix}\begin{bmatrix} 1 \\ x_1\\x_2  \end{bmatrix} = -3 + x_1+x_2$이다.

$\theta^Tx = -3+x_1+x_2 \geq 0 $이면 , 결과적으로 $x_1+x_2 \geq 3$인 영역에 있는 데이터들은 y =1로 예측할 수 있다. 반대로 $x_1+x_2 < 3$이면 y = 0으로 예측할 수 있다. 직선 $x_1+x_2=3$로 나눠진 영역에 따라 데이터들을 예측할 수 있으므로 이를 Decision Boundary라고 한다.



<strong><u>Non-linear decision boundary</u></strong>

polonomial regress에서 고차항을 추가하는 것처럼 동일하게 logistic regression에도 $x_1^2, x_2^2$의 feature을 추가한다.

ex) $h_\theta(x) = g(\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_1^2+\theta_4x_2^2)$

파라미터 $\theta^T = \begin{bmatrix} -1 &0&0&1&1\end{bmatrix}$로 가정하자.

그렇다면 $z=\theta^Tx = \begin{bmatrix} -1 &0&0&1&1\end{bmatrix}\begin{bmatrix} 1 \\ x_1\\x_2\\x_3\\x_4  \end{bmatrix} = -1+x_1^2+x_2^2$이다.

$\theta^Tx = -1+x_1^2+x_2^2 \geq 0 $이면, 결과적으로 $x_1^2+x_2^2 \geq 1$인 영역에 있는 데이터 들은 y = 1로, 반대로  $x_1^2+x_2^2 < 1$인 영역에 데이터들은 y=0으로 예측할 수 있다

이와 같이 고차다항식을 추가하면 더욱 복잡한 Decision Boundary를 얻을 수 있다.



Decision Boundary는 training set으로부터 결정된 파라미터 $\theta$에 의해 완전히 정의된다. 따라서 Decision Boundary는  training set과는 상관없이  파라미터 $\theta$의 값을 포함하는 $h_\theta(x)$의 속성이다.



<strong><u>Cost function for logistic regression</u></strong>

Cost function : $J(\theta) = \frac{1}{m}\Sigma_{i=1}^m\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$

$\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2 = Cost(h_\theta(x^{(i)}),j^{(i)})$로 두면 결과적으로  $J(\theta) = \frac{1}{m}\Sigma_{i=1}^mCost(h_\theta(x^{(i)}),j^{(i)})$이다.



하지만 logistic regression에서 $h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$가 비볼록함수 이기때문에 이 sigmoid funtion을 적용한 $J(\theta)$는 많은 극소점을 가진 함수의 형태로 나타나게 된다. 이렇게 되면 gradient descent과정을 거치더라도 global minimum에 도달한다는 보장이 없게 된다. 따라서  비볼록함수가 아닌 볼록함수를 적용해야 올바르게 gradient descent가 가능하다.



$Cost(h_\theta(x),y) = \begin{cases}-log(h_\theta(x)) & \mbox{if }y=1 \\ -log(1-h_\theta(x)) & \mbox{if } y = 0\end{cases}   $ ($0 \leq h_\theta(x) \leq 1$)

x축은 $h_\theta(x)$, y는 우리가 세운 $h_\theta(x)$과 관련하여 실제 데이터(y)와 차이로 인해 발생하는 비용이다. 

만약  $y = 1$일 때,  $h_\theta(x)=1$이면 우리가 예측한 $h_\theta(x)$가 실제 데이터를 정확하게 예측했으므로 Cost = 0이지만 $h_\theta(x)=0$이면 실제 데이터를 정확하게 예측하지 못했으므로 그 비용은 무한대로 커지게 된다.

반대로 $y = 0$일 때, $h_\theta(x)=0$이면 정확하게 예측했으므로 Cost = 0이지만 $h_\theta(x)=1$이라면 비용이 무한대로 커지게 된다는 것을 알 수 있다.



<strong><u>Simplified cost function and gradient descent</u></strong>

cost function을 더 간단하게 쓰는 방법

classification에서 training set의 y 값은 0 또는 1의 값만 가질 수 있기 때문에 위 $Cost(h_\theta(x),y)$식처럼 y=1, y=0의 경우로 나누어 표현하지 않고 하나의 식으로 압축하여 표현할 수 있다.

$Cost(h_\theta(x),y) = -ylog((h_\theta(x))-(1-y)log(1-h_\theta(x))$

따라서 하나로 표현된 $Cost(h_\theta(x),y)$를 $J(\theta)$에 대입하면 다음과 같이 표현할 수 있다.

$J(\theta) = \frac{1}{m}\Sigma_{i=1}^mCost(h_\theta(x^{(i)}),j^{(i)})= -\frac{1}{m}[\Sigma_{i=1}^m(y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})))]$



logistic regression cost function을 최소화하기 위해서는 gradient descent를 사용한다.

gradient descent를 위해 $J(\theta)$를 미분하면 다음과 같은 식이 생성된다.

Repeat { 

$\theta_j := \theta_j - \alpha\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

 }

이 식은 결과적으로 linear regression algorithm과 일치한다는 것을 알 수 있다. 다른 점은 hypothesis의 정의가 $h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$로 바뀌었다는 것 뿐이다.

우리가 logistic regression을 구현할 때 $\theta_0$부터 $\theta_n$까지 모든 파라미터들이 동시에 update되도록 구현해야 한다. 이는 0부터 n까지 for loop를 하거나, vertor rise implementation을 사용하면 된다.



<strong><u>Advanced optimization</u></strong>

logistic regression에서 cost function을 최소화하기 위한 advanced concept

gradient descent는 파라미터 $\theta$ 를 입력받아서 $J(\theta)$와 ${\partial \over\partial\theta_j}J(\theta)$ 를 계산하는 코드를 작성하고 이것을 적용하여  순차적으로 $\theta_j$를 update하는 과정을 반복해야만 했다.

gradient descent를 대신하여 위의 과정보다 더 최적화된 알고리즘은 Conjugate gradient, BFGS, L-BFGS가 있다. 



advanced optimization을 적용하는 방법

ex) $J(\theta) = (\theta_1-5)^2+(\theta_2-5)^2$, $\theta =\begin{bmatrix} \theta_1 \\ \theta_2\end{bmatrix} $

function [jval, gradient] = costFunction(THETA);

$jval = (\theta_1-5)^2+(\theta_2-5)^2$; $gradient = zeros(2,1)$;

$gradient(1) = 2*(\theta_1-5)$; $gradient(2) = 2*(\theta_2-5)$; -> octave에서는 index가 1부터 시작한다.

1. function [jval, gradient] = costFunction(THETA)

   -> $\theta$값을 입력받아 jval에 $J(\theta))$값을 저장하고 gradient에는 각각 파라미터 $\theta_n$에 대한 $J(\theta)$의 편미분 값을 저저장한다.

2. advanced algorithm 호출

   options = optimset('GradObj', 'on', 'MaxIter', '100') ->데이터구조의 옵션제공

   ​	(이것은 최대반복횟수를 100으로 설정하고 GradObj변수가 on으로 설정된다는 의미)

   initialTheta = zeros(2,1) -> $\theta$값 초기화

   [opTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, option) ->알고리즘 실행

장점 

1. $\alpha$를 따로 구할 필요가 없다. 왜냐하면 이 알고리즘 안에 수많은 $\alpha$를 적용해보고 알맞은 값을 선택하는 루프가 있기 떄문이다.
2. gradient descent보다 빠르다.
3. 라이브러리를 사용할 수 있다.

단점

1. gradient descent보다 복잡하다.
2. 디버깅하기 어렵다.  



<strong><u>Multiclass classification problems</u></strong>

Multiclass calssification에 대한 logistic regression은 one vs all algorithm을 사용한다.

예를 들어, 3개의 class가 있는 data set이 주어졌다면, one vs all로 binary class와 동일하게 적용한다.

하나의 class는 그대로 두고 다른 2개의 class를 새로운 하나의 training set으로 만들어 binary class를 적용한다.



ex) Triangle set = 1, square set = 2, crosses set = 3

triangle vs squares and crosses  -> $h_\theta^1(x)$, $P(y=1|x_1;\theta)$

squares vs triangle and crosses  -> $h_\theta^2(x)$, $P(y=2|x_2;\theta)$

crosses  vs triangle and squares  -> $h_\theta^3(x)$, $P(y=3|x_3;\theta)$

=>$h_\theta^{(i)}(x) = P(y=i|x;\theta)$,  (i = 1,2,3)

여기서 새로운 input x가 주어진다면, 3개의 $h_(x)$에 대해 모두 계산을 해보고 최대값이 나온 class i를 고르면 된다.