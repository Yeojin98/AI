<strong><u>Neural network cost function</u></strong>

<u>Types of classification problems with NNs</u>

$L$ : number of layers in the network

$s_l$ : number of units in layer $l$ -> not counting bias unit

1. binary classification : single output node($y \in \real$) , output layer $k=1$, $s_L=1$

2. multi-class classification: output node($y \in \real^k$), output layer $k \geq 3$, $s_L = k$

   multi-class는 k개의 class들을 classification하는 것이다. k가 2개 이하가 되면 one vs all method를 이용할 필요 없이 binary classification이 되므로 k는 항상 3이상이어야한다.

 <u>cost function for neural networks</u>

1. logistic regression cost function

   $J(\theta) = -\frac{1}{m}[\Sigma_{i=1}^my^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})]+\frac{\lambda}{2m}\Sigma_{j=1}^n\theta_j^2$

2. neural networks 

   $h_\theta(x) \in \real^K$, $(h_\theta(x))_i = i^{th}$ output ($h_\theta(x)$ 의 k dimensional vector에서 $i$번째 value)

   $J(\theta)= -\frac{1}{m}[\Sigma_{i=1}^m\Sigma_{k=1}^Ky_k^{(i)}log(h_\theta(x^{(i)}))_k+(1-y_k^{(i)})log(1-(h_\theta(x^{(i)}))_k)]+\frac{\lambda}{2m}\Sigma_{l=1}^{L-1}\Sigma_{i=1}^{s_l}\Sigma_{j=1}^{s_{l+1}}(\theta_{ji}^{(l)})^2$

   $\Sigma_{k=1}^K$는 각 K개의 output node에 대한 합계이다. 

   예를 들어 final layer의 node가 4개이고 training set이 m개라면, training set 1개로 각 output node마다 총 4번 summation이 발생된다. 이것이 반복되어 총 4m번의 summation이 발생되는 것이다.

   첫항 : $-\frac{1}{m}[\Sigma_{i=1}^m\Sigma_{k=1}^Ky_k^{(i)}log(h_\theta(x^{(i)}))_k+(1-y_k^{(i)})log(1-(h_\theta(x^{(i)}))_k)]$

   -> training example에 대한 각 output vector의 sum

   

   두번째 항: $\frac{\lambda}{2m}\Sigma_{l=1}^{L-1}\Sigma_{i=1}^{s_l}\Sigma_{j=1}^{s_{l+1}}(\theta_{ji}^{(l)})^2$-> 정규화

   -> 세 개의 중첩된 Sigma로 구성된 regularization summation term으로 weight decay term이라고 불린다. 

   $\Sigma_{l=1}^{L-1}$ : layer와 layer 사이를 mapping 해주는 파라미터들을 정규화 하기 위한 것으로 layer 1은 input layer이기 때문에 $L-1$만큼만 반복한다.

   $\Sigma_{i=1}^{s_l}$,$\Sigma_{j=1}^{s_{l+1}}$ : layer과 layer 사이를 mapping하는 파라미터의 dimension은 $[s_{j+1} \ \text{by}\ s_j+1]$이므로 $\Sigma_{i=1}^{s_l}$는 현재 layer의 node의 개수,  $\Sigma_{j=1}^{s_{l+1}}$ 다음 layer의 node개수이다.



<strong><u>Summary of what's about to go down</u></strong>

1. **forward propagation** 

   neural network에 initial input을 이용하여 단계적으로 network의 layer에 입력값을 push하여 $h_\theta(x)$를 도출binary classification의 경우 $h_\theta(x)$가 실수일 수 있고 vector일 수 있다.

2. **back propagation**

   network에서 얻은 output을 실제 training set의 y값과 비교하여 network의 파라미터가 얼마나 잘못되었는 지의 error를 계산한다. 

   이 error를 이용하여 이전 layer의 각 unit와 관련된 error들을 역으로 계산한다. 

   (input layer에는 입력값일 뿐이므로 error가 없기 때문에 input layer에 도달할 때까지 반복한다. )

   각 unit에서 측정한 error($\delta$)나 chain rule을이용한 ${\partial \over\partial\theta_{ij}^{(l)} }$로 를 계산한다. 

   gradient descent와 편도함수로 cost function을 최소화하고 모든 파라미터 값을 없데이트 한다.
   
   gradient descent가 수렴할 때까지 이 과정을 반복한다. 



ex) input layer(layer 1) :$x_1$,$x_2$ layer 2 : $a_1^2$,$a_2^2$,$a_3^2$ layer 3 : $a_1^3$,$a_2^3$인 NN

forward propagation

1. $z_1^2 = \theta_{11}^1x_1 + \theta_{12}^1x_2$ -> $g(z_1^2)=a_1^2$

   $z_2^2 = \theta_{21}^1x_1 + \theta_{22}^1x_2$ -> $g(z_2^2)=a_2^2$

   $z_3^2 = \theta_{31}^1x_1 + \theta_{32}^1x_2$ -> $g(z_1^3)=a_3^2$

2. $z_1^3 = \theta_{11}^2a_1^2 + \theta_{12}^2a_2^2+\theta_{13}^2a_3^2$ -> $g(z_1^3)=a_1^3$

   $z_2^3 = \theta_{21}^2a_1^2 + \theta_{22}^2a_2^2+\theta_{23}^2a_3^2$ -> $g(z_2^3)=a_2^3$

back propagation

loss function = $ylogh_{(x)} +(1-y)log(1-h_{(x)})$ 

2. $\theta_{11}^2$

    ${\partial{E_{total1}} \over\partial\theta_{11}^{(2)} } = {\partial{E_{total1}} \over\partial{a_{1}^{(3)}} } {\partial{a_{1}^{(3)}} \over\partial{z_{1}^{(3)}} } {\partial{z_{1}^{(3)}} \over\partial{\theta_{11}^2} }$

   $\theta_{12}^2$

    ${\partial{E_{total1}} \over\partial\theta_{12}^{(2)} } = {\partial{E_{total1}} \over\partial{a_{1}^{(3)}} } {\partial{a_{1}^{(3)}} \over\partial{z_{1}^{(3)}} } {\partial{z_{1}^{(3)}} \over\partial{\theta_{12}^2} }$

   $\theta_{13}^2$

    ${\partial{E_{total1}} \over\partial\theta_{13}^{(2)} } = {\partial{E_{total1}} \over\partial{a_{1}^{(3)}} } {\partial{a_{1}^{(3)}} \over\partial{z_{1}^{(3)}} } {\partial{z_{1}^{(3)}} \over\partial{\theta_{13}^2} }$

   $\theta_{21}^2$

    ${\partial{E_{total2}} \over\partial\theta_{21}^{(2)} } = {\partial{E_{total2}} \over\partial{a_{2}^{(3)}} } {\partial{a_{2}^{(3)}} \over\partial{z_{2}^{(3)}} } {\partial{z_{2}^{(3)}} \over\partial{\theta_{21}^2} }$

   $\theta_{22}^2$

    ${\partial{E_{total2}} \over\partial\theta_{22}^{(2)} } = {\partial{E_{total2}} \over\partial{a_{2}^{(3)}} } {\partial{a_{2}^{(3)}} \over\partial{z_{2}^{(3)}} } {\partial{z_{2}^{(3)}} \over\partial{\theta_{22}^2} }$

   $\theta_{23}^2$

    ${\partial{E_{total2}} \over\partial\theta_{23}^{(2)} } = {\partial{E_{total2}} \over\partial{a_{2}^{(3)}} } {\partial{a_{2}^{(3)}} \over\partial{z_{2}^{(3)}} } {\partial{z_{2}^{(3)}} \over\partial{\theta_{23}^2} }$

   

3. $\theta_{11}^1$

    ${\partial{E_{total}} \over\partial\theta_{11}^{(1)} } =({\partial{E_{total1}} \over\partial{a_{1}^{(2)} }}+{\partial{E_{total2}} \over\partial{a_{1}^{(2)} }}) {\partial{a_{1}^{(2)}} \over\partial{z_{1}^{(2)}} } {\partial{z_{1}^{(2)}} \over\partial{\theta_{11}^1} }$

   ${\partial{E_{total1}} \over\partial{a_{1}^{(2)} }} = {\partial{E_{total1}} \over\partial{a_{1}^{(3)} }}{\partial{a_{1}^{(3)}} \over\partial{z_{1}^{(3)} }}{\partial{z_{1}^{(3)}} \over\partial{a_{1}^{(2)} }}$

   ${\partial{E_{total2}} \over\partial{a_{1}^{(2)} }} = {\partial{E_{total2}} \over\partial{a_{2}^{(3)} }}{\partial{a_{2}^{(3)}} \over\partial{z_{2}^{(3)} }}{\partial{z_{2}^{(3)}} \over\partial{a_{1}^{(2)} }}$

   $\theta_{11}^1 = \theta_{11}^1-\alpha{\partial{E_{total}} \over\partial\theta_{11}^{(1)} }$

이런식으로 chain rule을 반복하여 모든 파라미터를 한꺼번에 update한다.



<u>What is back propagation?</u>

layer l에서 j번째 노드의 error를 나타내는 $\delta_j^l$을 계산한다. error는 real value인 y값과 activation되어 계산된 값과의 차이를 의미한다. 

[Activation of the unit] - [the actual value observed in the training set]

$\delta_1^3 ={\partial{E_{total}} \over\partial{z_{1}^{(3)}} } =a_1^3-y $

증명 )${\partial{E_{total}} \over\partial{z_{1}^{(3)}} } = {\partial{E_{total}} \over\partial{a_{1}^{(3)}} }{\partial{a_{1^{(3)}}} \over\partial{z_{1}^{(3)}} } = {\partial{E_{total}} \over\partial{a_{1}^{(3)}} }g'(z_1^3)= (\frac{y}{a_1^{(3)}}+\frac{(y-1)}{1-a_1^{(3)}})g'(z_1^3) = \frac{y-ya_1^{(3)}+ya_1^{(3)}-a_1^{(3)}}{a_1^{(3)}(1-a_1^{(3)})}a_1^{(3)}(1-a_1^{(3)}) = y-a_1^{(3)}$

$\delta_1^2 = {\partial{E_{total}} \over\partial{z_{1}^{(2)}} }={\partial{E_{total}} \over\partial{z_{1}^{(3)}} }{\partial{z_{1}^{(3)}} \over\partial{a_{1}^{(2)}} }{\partial{a_1^{(2)}} \over\partial{z_{1}^{(2)}} } = \delta^3(\theta^{(2)})^Tg'(z_1^2)$



$\delta^{(2)} = (\theta^{(2)})^T\delta^{(3)}.*g'(z^{(2)})$ 여기서 .*는 행렬간의 곱셈을 나타내는 기호이다.

$\delta^{(1)} = (\theta^{(1)})^T\delta^{(2)}.*g'(z^{(1)})$



$g(z^{(2)}) = a^{(2)}$이므로 $g'(z^{(2)}) = a^2 .*(1-a^2)$이다.

따라서, $\delta^2 = (\theta^2)^T\delta^3.*(a^2.*(1-a^2))$로 쓸 수 있다.



더 간단하게는,

 ${\partial{E_{total}} \over\partial\theta_{ij}^{(l)} } = {\partial{E_{total}} \over\partial{a_{i}^{(l+1)}} } {\partial{a_{i}^{(l+1)}} \over\partial{z_{i}^{(l+1)}} } {\partial{z_{i}^{(l+1)}} \over\partial{\theta_{ij}^l} } = {\partial{E_{total2}} \over\partial{z_{i}^{(l+1)}} }{\partial{z_{i}^{(l+1)}} \over\partial{\theta_{ij}^l} }  = \delta_i^{(l+1)}a_j^l$이므로  ${\partial \over\partial\theta_{ij}^{(l)} } = a_j^l\delta_i^{(l+1)}$이라는 것을 알 수 있다.따라서 위에서 말했듯이, 각 unit에서 측정한 error로 gradient descent를 위한 편도 함수를 계산 할 수 

있다. 



ex) More complex example

training set : ${\{(x^{(1)},y^{(1)}), \cdots,(x^{(m)},y^{(m)}) }\}$

1. set the delta values

   $\Delta_{ij}^{(l)} = 0 $ (for all $l$,$i$,$j$) 

   $\delta(l+1)$은 $\delta(l)$를 계산하기 위해 쓰이기 때문에 이 값들을 저장할 matrix 역할을 하게 된다.

2. loop through the training set

   training set $(x^{(i)},y^{(i)})$ 를 m 번 루프로 반복하여  학습시킨다.

   첫번째 루프에서 $a^{(1)} = x^{(i)}$이 저장되고 각각의 layer들에 $a^l\ (l=1,2,\cdots,L)$을 계산하기 위해 forward propagation을 수행한다.

   그 후에 output layer에서 $\delta^{(L)} = a^{(L)}-y^{(i)}$를 계산하고 back propagation을 이용하여 layer L-1까지의 $\delta$값을 구한다. 여기서 각 노드마다 $\delta$를 저장하기 위해 하기 위해  $\Delta_{ij}^{(l)} = \Delta_{ij}^{(l)}+a_j^l\delta_i^{(l+1)}$를 이용한다.  $\Delta^{(l)} = \Delta^{(l)}+\delta^{(l+1)}(a^{(l)})^T$가 되어 모든 i,j에 대하여 자동으로 업데이트 되는 $\delta$ matrix를 구현할 수 있다.

   

3. finally, exit the loop

   $D_{ij}^{(l)} := \frac{1}{m}\Delta_{ij}^{(l)}+\lambda\theta_{ij}^{(l)}$ if $j \ne 0$

   $D_{ij}^{(l)} := \frac{1}{m}\Delta_{ij}^{(l)}$ if $j = 0$ ->  bias unit

   $D$는 실수이고  ${\partial \over\partial\theta_{ij}^{(l)} }= D_{ij}^{(j)}$이다.




