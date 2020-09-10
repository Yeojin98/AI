<strong><u>Linear regression with multiple features</u></strong>

하나 이상의 변수나 features를 가진 선형 회귀

n = feature의 개수, m = examples의 개수, $x^i$= i번째 입력 벡터, ${x_j}^i$ = i번째 training exapmles의 j번째 feature



multiple features를 가진 hypothesis $h_\theta(x)$ = $\theta_0x_0+\theta_1x_1+\theta_2x_2 +...+\theta_nx_n$ ($x_0 =$1)일 때, 

$x = \begin{bmatrix} x_0\\x_1\\x_2\\ \vdots \\x_n\end{bmatrix}$$\theta = \begin{bmatrix} \theta_0\\\theta_1\\\theta_2\\ \vdots \\\theta_n\end{bmatrix}$이므로 $h_\theta(x)=\theta^Tx$로 표기할 수 있다.



<strong><u>Gradient Descent for Miltiple variables</u></strong>

1. Hypothesis : $h_\theta(x)=\theta^Tx= \theta_0x_0+\theta_1x_1+\theta_2x_2 +...+\theta_nx_n$
2. parameters : $\theta^T \begin{bmatrix} \theta_0, \theta_1 \cdots \theta_n \end{bmatrix}$
3. cost fuction : $J(\theta_0, \theta_1 \cdots \theta_n ) =J(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})^2$
4. Gradient descent : Repeat  {$\theta_j := \theta_j-\alpha{\partial \over\partial\theta_j}J(\theta_0, \theta_1 \cdots \theta_n)$}

Gradient descent하며 $\alpha{\partial \over\partial\theta_j}J(\theta_n)$가 계속 덮어 씌워지면서 $\theta_j$가 계속 update된다.



i) n = 1일 경우,

Repeat {

 $\theta_0 := \theta_0-\alpha \frac{1}{m}\Sigma_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})1 = \theta_0 -\alpha{\partial \over\partial\theta_j}J(\theta)$

$\theta_1 := \theta_1-\alpha \frac{1}{m}\Sigma_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})x^{(i)}$

}

ii) $n\geq 1$인 경우,

 Repeat  {

$\theta_j := \theta_j-\alpha\frac{1}{m}\Sigma_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})x_j^{(i)}$

}

 위의 두 경우를 살펴보았을 때 공통적으로 $\frac{1}{m}\Sigma_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})x_j^{(i)}$이 부분이 포함되어있으므로 n이 1이던 1보다 크던 Gradient Descent for Multiple variables를 이용하여 같은 결과값을 얻을 수 있게 된다. 



<strong><u>Gradient Descent in practice : 1 Feature Scaling</u></strong>

 만약 여러 개의 feature가 있고, feature의 단위크기가 서로 비슷하다면 gradient descent는 더 빠르게 수렴하지만 그렇지 않고 단위크기의 차이가 크다면 $\theta_n$ 파라미터에 따른 $J(\theta)$의 그래프가 매우 뾰족한 타원모양이 나오게 된다. 이러한 뾰족한 타원모양은 gradient descent가 앞뒤로 진동하며 오랜시간이 지나고서야 global minimum에 도착하게 되므로 매우 비효율적이다.

=> 따라서, feature간의 단위크기 차이가 클 경우 더 효율적인 gradient descent를 위해 input들을 rescale할 필요가 있다.

=> Mean nomalization 

: 각 feature들에 대해 평균값을 빼고 최댓값으로 나누어 feature들의 값을 -1과 1 사이의 비율로 조정

$x_i$ <- $\frac{x_i - (\mu_i)}{S_i} $      ($\mu_1$ = traing set에 있는 $x_i$값들의 평균,   $S_i$=단위크기의 차이)



<strong><u>Learning rate</u></strong>

 Gradient descent의 목적은 cost function $J(\theta)$가 최소화되는 $\theta$값을 찾는 것이다. 따라서 Gradient descent가 올바르게 작동하고 있다면 $J(\theta)$가 매번 감소해야한다. 이는 직접 반복 횟수에 대한 cost function을 그려보면 $J(\theta)$가  감소하는지, 어느 반복 횟수에 수렴하는지 알 수 있다.

 하지만 사전에 Gradient descent가 수렴하기 위해서 몇번을 반복해야하는지 알기 어렵기 때문에 Automatic convergence test를 사용한다.  Automatic convergence test는 어떤 임계값을 정하고 $J(\theta)$가 이 임계값의 이하로 변경되는지 확인하는 방식으로 작동된다. 이것은 임계값을 결정하기도 어렵다는 단점이 있다.

 $J(\theta)$값이 점점 커지는 경우에는 더 작은 learning rate를 사용해야한다. 예를 들어 함수에 따라  이미 $J(\theta)$가 최솟값에 가까운 경우에 learning rate가 너무 크다면 최솟값을 지나쳐  점점 높은 값을 가지게 될 수 있다. 

 주로 $J(\theta)$가 감소하지 않는다면 learning rate를 감소시키면 되지만 너무 작아져서 Gradient descent가 천천히 수렴하는 것을 피하는 것이 좋으므로 적당히, 적절하게 작은 learning rate를 찾아야한다.



<strong><u>Features and polynomial regression</u></strong>

 여러개의 feature이 있을 때, Gradient descent를 효율적으로 하기 위한 feature들을 선택하는 방법이 필요하다.

1. 여러 개의 feature들의 연산으로 새로운 feature을 만든다.

   ex) 집값을 예측하는 예제에서 frontage, depth 이 두 가지의 feature이 있다면? 

   frontage와 depth를 곱하여 Area라는 새로운 feature를 만들어 사용하는 것이 효율적이다.

2. Polynomial regression

   : data set을 나타내기 위한 $h_\theta(x)$가 직선 함수보다 다항함수가 효율적일 떄, multivarient linear regression의 구조를 이용하여 $h_\theta(x)$를 feature들로 나타내는 방식이다.

   ->  $h_\theta(x)$의 차수가 커질 때마다 변수들의 단위크기차이가 커지므로 feature scaling을 적용시켜 비슷한 범위를 가지도록 하는 과정이 중요하다.

   -> 다항식에 제곱근 세제곱근 등의 변수를 사용할 수 있어 데이터를 효율적으로 나타낼 수 있다.



<strong><u>Normal equation</u></strong>

 지금까지 Linear Regression에 사용한 알고리즘은 Gradient descent이고,  $J(\theta)$가 최솟값에 수렴하도록 Gradient Descent를 여러번 반복하는 과정을 거쳤다. 이 점을 보완하여  한번의 실행으로 $\theta$의 최적의 값을 한번에 구할 수 있는 것이 Normal equation이다.

 $J(\theta_0, \theta_1 \cdots \theta_n ) = \frac{1}{2m}\sum_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})^2$ , $(\theta \in \mathbb{R}^{n+1})$일 때, ${\operatorname{d}\over\operatorname{d}\!\theta_j}J(\theta)$가 모든 j에 대하여 모두 0이 되도록 하면 모든 $\theta$값에 대한 답을 구할 수 있게 되고 이것을 cost function에 각각 대입하면 $J(\theta)$를 최소화할 수 있다. 하지만 이 방법은 연산이 꽤 어렵고 시간이 오래 걸린다는 단점이 있다. 

 normal equation : 하나하나 미분하는 대신 행렬을 이용하여 $\theta$를 구한다.

ex)

| size$(feet^2)$ | NUmber of bedrooms | Number of floors | Age of home(years) | price($1000) |
| -------------- | ------------------ | ---------------- | ------------------ | ------------ |
| $x_1$          | $x_2$              | $x_3$            | $x_4$              | $y$          |
| 2104           | 5                  | 1                | 45                 | 460          |
| 1416           | 3                  | 2                | 40                 | 232          |
| 1534           | 3                  | 2                | 30                 | 315          |
| 852            | 2                  | 1                | 36                 | 178          |

이 표에서 $x_0$ column을 모두 1로 하여 추가한다음 행렬을 구성한다.

$X = \begin{bmatrix} 1&2104&5&1&45 \\ 1&1416&3&2&40\\1&1534&3&2&30\\1&852&2&1&36 \end{bmatrix}$ $Y = \begin{bmatrix} 460 \\ 232\\315\\178 \end{bmatrix}$ X는 (m x n+1)차원, Y는 (m x 1)차원 벡터이다.

여기에서 $\theta = (X^TX)^{-1}X^Ty$을 계산하면 $J(\theta)$를 최소화하는 $\theta$를 한번에 구할 수 있게 된다.



=> 이것을 일반화하면, m개의  $(x^{(1)}, y^{(1)}) \cdots (x^{(n)}, y^{(n)})$ examples가 있을 때,

$X$(Design matrix) = $\begin{bmatrix} (x^{(1)})^T \\ (x^{(2)})^T\ \\\vdots\\(x^{(m)})^T \end{bmatrix}$이 되어 (m x n+1)차원이 되고, Y= $\begin{bmatrix} (y^{(1)})\\ (y^{(2)})\ \\\vdots\\(y^{(m)})\end{bmatrix}$ (m x 1)차원에 벡터를 만들게 된다.  

X와 Y행렬을 통해 $\theta = (X^TX)^{-1}X^Ty$를 연산한다. 

MATLAB이나 octave에서 prinv($X'*x$)*x'*y로 쉽게 구 할 수 있다.



When should you use gradient descent and when should you use feature scaling?

1. Gradient descsnt -> feature scaling이 필요하다.

   장점 : feature가 많을 떄 효과적이다.

   단점 :  learning rate 즉, $\alpha$를 정해야한다. 

   다양하게 $\alpha$를 바꿔가며 gradient descent를 해야하기 떄문에 반복이 많다.

2. normal equation -> feature scaling이 필요없다.

   장점 : $\alpha$를 정하지 않아도 된다. 구현이 간단하다. 반복도 없다. 

   단점 : feature가 많아지면 $\theta$를 연산하기가 매우 복잡하다. feature이 많아질수록 계산속도가 세제곱배로 느려진다.

=> n이 엄청 클 때 ($n \geq 10^4$), 즉 feature가 너무 클 때는 주로 Gradient descent를 사용하고 

​	n이 작을 떄는 nornal equation을 이용하는 것이 좋다.

