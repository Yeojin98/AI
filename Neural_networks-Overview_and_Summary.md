<Strong><u>Neural networks - Overview and Summary</u></strong>

feature가 1~2개인 경우 복잡하게 분포되어있는 데이터들에 대해  다항식을 이용하여 classification할 수 있겠지만, feature가 100개인 경우에는 모든 이차항을 포함하는 항의 개수가 5000개가 된다. 

feature의 수가 많아질 수록 이차항의 개수도 $O(n^2)$만큼 늘어나게 되어 계산 비용이 많이 들게 된다. 또한 모든 이차항을 포함하면 traing set에 딱 맞는 cost function이 만들어져 overfitting이 발생할 수 있다.

따라서 모든 이차항들을 포함하기보다는 feature의 subset만을 포함하도록 하는 것이다. 이런 방식은 모든 data set을 가려내지는 못하지만 feature가 늘어남에 따라 발생하는 feature space를 생각하면 훨씬 효율적이다.



따라서, feature가 지나치게 많은 경우 feature space가 너무 커지므로  비효율적이므로 Neural networks를 사용하는 것이 훨씬 효과적이다.



Nearal networks : 인간의 뇌의 기능를 모방하는 알고리즘

뇌는 single learning algorithm을 가지고 있다고 가정한다.

소리를 듣는 청각피질을 귀가 아닌 눈에 연결하면 청각피질은 무언가를 보는 방법을 배우고 촉감을 느끼는 somatosensory context를 눈에 연결하면 시신경과 같이 무언가를 보는 방법을 배운다.

따라서, 뇌는 모든 소스의 데이터를 처리하고 학습할 수 있는 능력이 있다.



<Strong><u>Model representation 1</u></strong>

Neural Networks는 뉴런의 네트워크를 simulate하는 것을 모방한다. 

뉴런의 구성 : Cell Body(세포체), Dendrite (수상돌기) 입력 단자 역할, Axon(축삭돌기) 출력 단자 역할

뉴련의 동작 : 뉴런은 여러개의 입력 단자로부터 값을 받아 특정 계산을 수행하고 축삭돌기를 통해 출력값을 spikes를 일으켜 다른 뉴런으로 보낸다.



<u>Artificial neural network - representation of a neurone</u>

 $x = \begin{bmatrix} x_0\\x_1\\x_2\\x_3\end{bmatrix}$=>입력 =>뉴런 => $h_\theta(x)$ => $\theta = \begin{bmatrix} \theta_0\\\theta_1\\\theta_2\\\theta_3\end{bmatrix}$

여기서, $x_0$는 **bias unit**으로 주로 1의 값을 가지고 $h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$ 로 sigmoid이다.

Layer 1에 입력 단자가 $x_1,x_2,x_3$, Layer 2에 뉴런 $a_1^{(2)},a_2^{(2)},a_3^{(2)}$, Layer 3에 layer 2의 결과값을 합산하여 $h_\theta(x)$ 도출하는 뉴런이 있다고 하자.

Layer 1을 **input layer**, Layer 3을 **output layer**, Layer 2을 **hidden layer**라고 한다. hidden layer는 supervised learning에서 input과 output은 볼 수 있지만 이 hidden layer는 직접 볼 수가 없기 때문에 hidden이라고 이름을 붙인것이다. 결과적으로 input과 output이 아닌 것은 모두 hidden layer이다.

<u>Neural networks-notation</u>

$a_i^{(j)}$ : activation of unit $i$ in Layer $j$

$\theta^{(j)}$ : matrix of parameters controllling the function mapping from layer $j$ to layer $j+1$

-> 파라미터는 하나의 layer에서 다음 layer로 mapping하는 control하는 matrix이다.

만약, $j$번째 layer가 $s_j$개의 unit을 가지고있고, $j+1$번째 layer가 $s_{j+1}$개의 unit을 가지고 있다면 $\theta^j$는 $[s_{j+1} \text{by}\ s_j+1]$ matrix가 될것이다.  

따라서, $\theta_j$의 column은 다음 layer의 수이고 Row는 현재 layer의 unit수에 bias unit 1을 더한 값이다.



ex) Layer 1에 $x_1,x_3,x_3$ 3개의 unit, Layer 2에 $a_1^2, a_2^2, a_3^2$ 3개의 unit, Layer 3에 $h_\theta(x)$ 1개의 unit이 있는  NN

$\theta^1$은 input units와 hidden unit을 mapping하는 $[3 \ \text{by}\ 4]$matrix 이고 $\theta^2$는 hidden layer와 output layer을 mapping하는  $[1 \ \text{by}\ 4]$ matrix이다.

가장 중요한 것은 모든 input이 다음 layer의 모든 node들에 연결된다는 것이다. 따라서 이 모든 layer transition을 나타낼 수 있는 $\theta$ matrix로 $\theta_{ji}^l$을 사용한다. $l$은 현재의 layer, $j$는 다음 layer의 j번째의 unit, $i$는 현재 layer에서 i번째 unit을 의미한다.



<Strong><u>Model representation 2</u></strong>

ex) Layer 1에 $x_1,x_3,x_3$ 3개의 unit, Layer 2에 $a_1^2, a_2^2, a_3^2$ 3개의 unit, Layer 3에 1개의 unit이 있는  NN

$a_1^{(2)} = g(\theta_{10}^{(1)}x_0+\theta_{11}^{(1)}x_1+\theta_{12}^{(1)}x_2+\theta_{13}^{(1)}x_3)$

$a_2^{(2)} = g(\theta_{20}^{(1)}x_0+\theta_{21}^{(1)}x_1+\theta_{22}^{(1)}x_2+\theta_{23}^{(1)}x_3)$

$a_3^{(2)} = g(\theta_{30}^{(1)}x_0+\theta_{31}^{(1)}x_1+\theta_{32}^{(1)}x_2+\theta_{33}^{(1)}x_3)$

$h_\theta(x)=a_1^{(3)} = g(\theta_{10}^{(2)}a_0^{(2)}+\theta_{11}^{(2)}a_1^{(2)}+\theta_{12}^{(2)}a_2^{(2)}+\theta_{13}^{(2)}a_3^{(2)})$

$\theta^{(1)}x= \theta_{10}^1x_0+\theta_{11}^1x_1+\theta_{12}^1x_2+\theta_{13}^1x_3=z_1^2$이라고 한다면, $a^{(2)} = g(z^{(2)})$이다. 

matrix로 표현하면  $x = \begin{bmatrix} x_0\\x_1\\x_2\\x_3\end{bmatrix}$  $z^{(2)} = \begin{bmatrix} z_1^{(2)}\\ z_2^{(2)}\\ z_3^{(2)}\end{bmatrix}$이다. 

이렇게 되면 $z$ matrix가 [3 by 1] matrix이고 $a^{(2)} = g(z^{(2)})$에 의해 $a$ matrix도 [3 by 1]matrix가 된다. 

하지만 우리가 결과적으로 구하고자 하는 $h_\theta(x)=a_1^{(3)} = g(\theta_{10}^{(2)}a_0^{(2)}+\theta_{11}^{(2)}a_1^{(2)}+\theta_{12}^{(2)}a_2^{(2)}+\theta_{13}^{(2)}a_3^{(2)})$에는 $a_0^{(2)}$을 연산해 줘야 하므로 $a_0^{(2)}$ = 1로 추가해줘야 한다. 따라서 $a$ matrix는 [4 by 1]  matrix이고 우리가 구하고자 하는 $h_\theta(x) = a^3=g(z^3)$와 같다.



input layer에서 $x$ vector로 activation을 시작하여 순차적으로 각각의 hidden layer를 거쳐 output layer에서 $h_\theta(x)$함수를 계산해내는 것을 **forward propagation**이라고 한다.



<u>Neural networks learning its own features</u>

ex) Layer 1이 가려져있고 Layer 2에 $a_1^2, a_2^2, a_3^2$ 3개의 unit, Layer 3에 1개의 unit이 있는  NN

$h_\theta(x)= g(\theta_{10}^{(2)}a_0^{(2)}+\theta_{11}^{(2)}a_1^{(2)}+\theta_{12}^{(2)}a_2^{(2)}+\theta_{13}^{(2)}a_3^{(2)})$이며 이 예제에서의 차이점은 input feature vactor 대신에 그 feature들이 hidden layer에서 계산된 것 즉, $a_1^2, a_2^2, a_3^2$ 을 이용하여 $h_\theta(x)$를 도출한다는 것이다.  여기서 $a_1^2, a_2^2, a_3^2$이 input layer를 대신하여 input layer의 기능을 학습한다. 따라서 원래의 input layer값에 제한을 받지 않고 스스로 학습하여 logistic regression가 가능하다.  logistic regression calculation에 필요한 feature들을 학습하는 유연성을 가지게 된다.



결과적으로 classification을 하기 위해 각각의 layer들이 스스로의 feature을 학습하여 최총 output layer에 input할 최상의 결과값을 도출하는 방법을 배우게 되는 것이다.



<Strong><u>Nearal network example - computing a complex, nonlinear function of the input</u></strong>

$x_1$,$x_2$는 binary이고 positive, negative examples들을 구분하는 nonlinear decision boundary

$(x_1,x_2) = (0,0),(1,1)$ 일때 negative, $(x_1,x_2) = (1,0),(0,1)$일 때 positive

$y = x_1 \text{XOR}x_2$은 $x_1$과 $x_2$ 다를 때 true를 반환하기 때문에 positive가 $y=1$ 된다. 보편적으로 negative를 $y=1$로 놓는 것과 반대되므로 $y = x_1 \text{XNOR}x_2$을 이용하는 것이 더 좋다.

1. Nearal Network example 1: AND Function

   -> $h_\theta(x) = g(-30+20x_1+20x_2)$

2. Nearal Network example 2: NOT Function

   -> $h_\theta(x) = g(10-20x_1)$

3. Nearal Network example 3: XNOR Function

   $a_1^{(2)} = g(-30+20x_1+20x_2)$

   $a_2^{(2)} = g(10-20x_1-20x_2)$

   $a_1^{(3)} = g(-10+20x_1+20x_2)$

   | $x_1$,$x_2$ | $a_1^{(2)}$,$a_2^{(2)}$ | $h_\theta(x)$ |
   | ----------- | ----------------------- | ------------- |
   | 0,0         | 0,1                     | 1             |
   | 0,1         | 0,0                     | 0             |
   | 1,0         | 0,0                     | 0             |
   | 1,1         | 1,0                     | 1             |

   

<Strong><u>Multiclass classfication</u></strong>

multiclass classfication은 output layer에 unit이 2개 이상 있는 것이다. 마찬가지로, one vs all의  calssification을 이용한다. 

ex) 차를 2가지의 카테고리로 나누어 보행자, 자동차. 오토바이, 트럭을 분류하는 예제

$y^i$는 해당 이미지의 분류를 나타내는 벡터, $x^i$는 4가지 분류 중 하나에 속하는 이미지

output layer에 길이가 4인 열벡터를 주고 보행자이면 $y =\begin{bmatrix} 1\\0\\0\\0\end{bmatrix}$이 되는것이다.

이미지의 분류를 예측하는 $h(x)$에 input인 $x^{(i)}$를 이용하면 $h(x^{(i)}) = y^i$가 되는 것이다.

