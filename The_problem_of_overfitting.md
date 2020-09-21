<strong><u>The problem of overfitting</u></strong>

1. Overfitting with linear regression

   ex) house pricing example

   1.  data에 linear function을 적용하는 것 -> underfitting, high bias

      단순한 feature만을 가진 직선으로 데이터를 fitting하여 bias하게 traing set을 나타나게 된다.

   2. 이차항 함수 -> works well

   3. 4차 다항식 함수 -> overfitting 

      training set에는 완벽히 맞는 것처럼 보이지만 너무 많은 변수를 사용해서 만든 좋지 않은 모델이다.

   => 많은 feature들을 가진 hypothesis는 cost function이 정확히 0의 값을 가지게 할 수 있다. 하지만 이것은 주어진 traning set에 완벽하게 맞아떨어지는 hypothesis이기 때문에 일반화하는 것에는 실패한다.

2. Overfitting with logistic regression

   linear regression과 같이 단순한 feature를 가진 sigmoid를 사용할 경우 underfitting, 너무 많은 feature를 가진 경우 overfitting이 된다.

=> Addressing overfitting

앞서 너무 높은 차수를 가진 함수들은 너무 "curvey"한 것을 확인했다. 이러한 경우 차수를 선택하는 것뿐만 아니라 어떤 feature를 사용할 건지 feature scaling을 시각화하는 것이 더 어렵다.

해결방법

1. Reduce number of features

   : 어떤 feature를 남길 건지 고른다. 이상적으로 데이터 손실을 최소화하는 feature을 선택하겠지만 그래도 일부의 정보는 손실될 수 밖에 없다.

2. Regularization

   : 모든 feature들을 남겨두되, 파라미터 $\theta$의 magnitide를 줄인다. Regularization은 feature이 많을 때 효과적이다.

<strong><u>Cost Function optimization for regularization</u></strong>

ex) $h_\theta(x)=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$

$\theta_3$와 $\theta_4$가 overfitting을 일으키는 주된 feature일 경우

$\theta_3$와$\theta_4$를 regularization하기 위해 엄청나게 큰 수의 상수를 가진 두 변수를 더해준다.

$min \frac{1}{2m} \Sigma_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+1000 \theta_3^2+1000\theta_4^2$

최종적으로  $\theta_3$와 $\theta_4$의 큰 계수로 인해 값이 계속 커지므로  위와 같은 cost function을 최소화하기 위해서는 $\theta_3$과 $\theta_4$를 0으로 수렴하게 해야한다. 

따라서, $J(\theta)= min \frac{1}{2m} \Sigma_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+1000 \theta_3^2(=0)+1000\theta_4^2(=0)$이므로 값이 변하지 않고 $h_\theta(x) = \theta_0+\theta_1x+\theta_2x^2$가 되어 완벽한 2차식이 된다.

=>이것을 일반화하면 파라미터들이 작은 값들로 이루어져 있다면, 이것은 더 간단한 $h_\theta(x)$를 만들 수 있다는 것이다.



ex) feature가 100개가 있는경우

다항식 예와 다르게 우리는 고차항을 가진 feature가 어떤 것인지 알 수 없다. 따라서, 어떤 feature을 남길 것인지, 어떤 feature을 줄일 것인지 결정하기 어렵다. 이러한 경우 $\theta_0$를 제외한 모든 파라미터를 줄인다. 파라미터의 값이 더 작을 수록 더 간단한 $h_\theta(x)$를 구할 수 있기 때문이다. 

$J(\theta) = \frac{1}{2m} [\Sigma_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\Sigma_{j=1}^{n}\theta_j^2]$ (여기서 $\theta_0$는 penalize할 필요가 없다.)

$\lambda$는 regularization parameter로 training set을 알맞게 표현해주면서 파라미터 값들을 작게 유지해주는 역할을 한다. $\lambda$가 매우 크면 모든 파라미터가 0에 가까워지고 training set을 직선으로만 표현이것은 underfitting와 high bias의 결과를 가져온다. 

따라서 $\lambda$는 너무 크지 않도록 신중하게 선택되어야 한다.   



<strong><u>Regularization for linear regerssion</u></strong>

$J(\theta) = \frac{1}{2m} [\Sigma_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\Sigma_{j=1}^{n}\theta_j^2]$

Gradient descent

Repeat{

$\theta_0 := \theta_0 - \alpha\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$ (j=0)

$\theta_j := \theta_j - \alpha[\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j]$  (j = 1,2,3 $\cdots$,n)

}

$\alpha[\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j]$이 항은 새롭게 정의된 $\alpha{\partial \over\partial\theta_j}J(\theta)$ 즉, $J(\theta)$에 대한 편도함수의 항이다.

따라서, $\theta_j := \theta_j(1-\alpha \frac{\lambda}{m}) - \alpha\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$ 로 묶어줄 수 있다.

첫 번째 항에서$(1-\alpha \frac{\lambda}{m})$의 의미는  $\theta_j$에 곱해져 파라미터 $\theta_j$값을 더 작게 만들어주어 overfiiting을 일으키는 파라미터값의 영향력을 작게 해준다.

두 번째 항에서 원래 gradient descent와 동일하게 동작한다.



<strong><u>Regularization with the normal equation</u></strong>

gradient descent를 반복하는 것보다 normal equation  $\theta = (X^TX)^{-1}X^Ty$를 이용하여 파라미터 $\theta$의 값을 한번에 연산하도록 한다.

$\theta = (x^Tx+\lambda \begin{bmatrix}0&0& \cdots& 0& 0\\ 0&1&\cdots &0&0 \\ \vdots&&\cdots&&\vdots\\ 0&0&\cdots&0&1 \end{bmatrix})^{-1}x^Ty$. (이 matrix는 0부터 n까지 [{n+1} * {n+1)] matrix 이다.)



<strong><u>Regularization for logistic regerssion</u></strong>

logstic regression에서 feature가 많아져 overfitting이 발생하는 경우 regularization을 하여

$J(\theta) = -\frac{1}{m}[\Sigma_{i=1}^m(y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})))]+\frac{1}{2m}\lambda\Sigma_{j=1}^{n}\theta_j^2$을 사용한다.

linear regeression에서와 마찬가지로 gradient descent를 할 때

$\theta_j := \theta_j - \alpha[\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j]$  (j = 0, 1,2,3 $\cdots$,n)

$\theta_j := \theta_j(1-\alpha \frac{\lambda}{m}) - \alpha\frac{1}{m}\Sigma_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$ 로 한다.

여기서, linear regression과 다른 점은 $h_\theta(x)$의 형태뿐이다.

