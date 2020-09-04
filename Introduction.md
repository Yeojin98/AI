<strong><u>**Introduction to the course **</u></strong>

•인공 지능의 목표는 인간만큼의 지능적인 기계를 만드는 것

• ML(machine learning)이 널리 퍼진 이유는?

1. 

   <ol>AI의 성장
   </ol>

2. <ol>지능형 기계 구축
       <ul>
           <br>
       간단한 조건을 가지는 프로그램들은 쉽게 코딩할 수 있지만 조건이 매우 많은 프로그램들은 프로그래머가 하나하나 코딩하기 어렵다.
       <br>
       </ul>
       <ul>
            따라서 가장 좋은 방법은 기계가 학습 메커니즘을 통해 스스로 학습할 수 있도록 하는 것이다.
       </ul>
   </ol>

<strong><u>**What is machine learning?**</u></strong>
$$
\centerdot \ Tom Michel(1999)
$$
“프로그램이 작업 성능(P)를 가지고 작업(T)를 수행한다고 했을 때, 경험(E)가 증가할 수록 작업(T)을 수행하는 성능(P)이 향상된다.”
$$
\centerdot \ Several\ types\ of\ learning\ algorithms
$$

1. <ol>
       supervised learning
       <ul>
           특정 input에 대해 정답이 되는 output이 있는 데이터 셋이 주어지는 경우, input과 output의 관계를 유추
           <ul>
               ex )　regression, classification
           </ul>
       </ul>
   </ol>

2. <ol>
       Unsupervised Learning 
       <ul>
           특정 input에 대해 정답이 되는 output이 없는 경우, prediction result와 feedback이 없음
           <ul>
               ex ) clustering
           </ul>
       </ul>
   </ol>

<strong><u>**Supervised learning **– **introduction**</u></strong>

1. ex ) Housing price prediction (Regression)
   • output이 연속적인 값을 갖고 이 값을 추정하는 문제

   <ol>
           주어진 데이터 셋에서  input과 output의 관계를 유추하여 원하는 input에 대한 특정 output을 도출해 내는 것이다.

   <ol>
   	이러한 관계로 구현된 알고리즘은 정확한 output을 알지 못하는 input에 대해 더 정	확한 값을 예측할 수 있도록 한다.


   <ul>
           => regression : 연속적인 값을 가진 결과를 예측, No discrete category
           </ul>

2.  ex ) 종양 크기에 따라 유방암을 악성 또는 양성으로 판단할 수 있는가 (Classification)

   • output이 discrete한 값을 가지는 문제

   <ol>
       출력에 대해 가질 수 있는 특정한 값을 정할 수 있다.
   </ol>

   <ol>
       종양 크기뿐만 아니라 환자의 나이를 input으로 활용하여 판단할 수 있다.
   	 이 알고리즘은 무한한 수의 특징들을 다룰 수 있다.
   </ol>

<strong><u>**Unupervised learning **– **introduction**</u></strong>

-> 분류되어있지 않은 데이터를 얻는다. 

<ol>
    이 데이터 세트를 구조화하기 위해서는 데이터를 그룹으로 클러스터링 하는 것이다. 
</ol>

1. Clustering algorithm => 비슷한 것끼리 묶는 것

    ex) microarray : 유전자의 발현을 측정할 때마다  개인을 특정한 유형으로 클러스터링

   <ol>
       -> 클러스터링 구성 : 잠재적인 취약점을 식별하거나 워크로드를 효과적으로 분배하도록
   </ol>

2. Cocktail party problem (non-clustering algorithm)

   : 마이크가 있는 위치에 따라 각각 다른 버전의 대화를 녹음

   <ol>
       알고리즘은 오디오 녹음을 2개의 소스로 분리한다.
   <ol>
   	-> 매우 복잡해 보이지만 octave나 MATLAB으로 한 줄로 코딩 가능
   <ol>		
       ◦[W,s,v] = svd((repmat(sum(x.*x,1), size(x,1),1).*x)*x');

<strong><u>**Linear regression - implementation (cost function)**</u></strong>

cost function은 데이터를 표현할 수 있는 가장 알맞은 직선을 알아낼 수 있도록 

$$h_\theta(x) = \theta_0+\theta_1x$$

\-> training set를 기반으로 직선을 만드는 파라미터들을 알아낸다.

<ol>
     training 예제에서 hθ(x)가 실제 y에 가까워지도록 하는 파라미터를 선택
</ol>
$$J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})^2$$

 우리가 예측하고자 하는 데이터를 표현하는 직선을 위와 같이 hθ(x)로 둔다. 여기에서 실제 training set의 y의 값과 차이를 최소화하는 θ0, θ1을 구하는 것이 cost function이다. 

 이를 위해 우리가 가정한 직선인 hθ(x)와 각각의 training set의 y 값과의 차이의 합이 가장 적은 것을 구하면 된다. 1/m은 각각 데이터들의 개수로 평균을 나타낸 것이다.

<strong><u>**cost function** **–** **a deeper look**</u></strong>

θ0 = 0이라고 두고, hθ(x) = θ1x 라 하자. training set은 (1,1), (2,2), (3,3)이라고 가정하자. 

여기에서 비용함수(J)를 최소화하는 θ1를 찾는다.

-> θ1=1, 일 때 J(θ1)=0. 

​    θ1=0.5일 때, J(θ1) = 0.58. 

​    θ1=0일 때, J(θ1)=2.3

 => θ1=1일 때 최솟값 0을 갖는 이차곡선 그래프가 그려진다.

<strong><u>**a deeper insight into the cost function** **–** **simplified cost function**</u></strong>

θ0와 θ1를 모두 사용하여 cost function의 그래프를 알아보자.

1. x축은 θ1, z축은 θ2, y축은 J(θ1,θ2)을 갖는 3D 형태의 그래프

   <ol>
        -> y축의 값이 가장 작은 곳을 찾는다.
   </ol>

2. 3D 그래프에서 x축은 θ1, y축은 θ2만으로 표현한 다양한 타원 그래프 

   <ol>
        -> 같은 색 타원 그래프는 갖은 cost funtion의 값을 가지지만 θ0, θ1은   다른 값을 가진다. 여기에서 타원 그래프의 중심으로 갈수록 cost function  이 작아진다는 것을 알 수 있다.
       </ol>

   <ol>
         등고선 그래프를 상상하여 기울기가 가파를수록 타원과 타원 사이에 간격이   좁고 기울기가 완만할수록 타원과 타원 사이에 간격이 점점 넓어진다. 
   </ol>

<strong><u>**gradient descent algorithm**</u></strong>

1. θ0와 θ1을 0으로 초기화

2. θ0와 θ1를 조금씩 바꾸며 cost function을 가장 작게 만드는 기울기를 찾는다.

3. 이를 반복하면 특정 시작점에 맞는 특정 local minimum을 알 수 있게 된다.

   <ol>
        -> 시작점이 다르면 cost funtion의 최솟값을 나타내는 local minimum도 다르다.
   </ol>
$$\theta_j := \theta_j-\alpha{\partial \over\partial\theta_j}J(\theta_0,\theta_1)$$  (for j= 0 and j=1)
   
   α는 learning rate로 α가 매우 크다면 가파른 기울기를 의미하고, α가 매우 작다면 완만한 기울기를 의미한다.
   
​    
   
1. 미분계수 : 특정 위치의 탄젠트 값 즉, 기울기를 구하는 방법이다. 
   
   <ol>
          기울기가 항상 양수이면 θj값이 작아져 θ1이 원점과 가까워지는 방향으로 최소  지점과 가깝게 이동한다. 
   </ol>
   
      <ol>
          기울기가 항상 음수이면 θj값이 커져서 θ1이 원점과 멀어지는 방향으로 최소 지점에  가깝게 이동한다.
   </ol>
   
   2. α 
   
   <ol>
          α가 매우 작으면 작은 수가 미분계수에 곱해져 최소 지점으로 조금씩만 움직여 최소  지점에 도달하기까지 많은 이동이 필요하다. 
   </ol>
   
      <ol>
           α가 너무 크면 큰 거리를 이동하게 되어 최소값에서 계속 멀어지게 된다.
   </ol>
   
      <ol>
          -> 만약 α가 고정되어있다면?
   </ol>
   
      <ol>
           최소 지점으로 향하는 그래프의 기울기 값이 점점 작아지기 때문에 α값이 고정되어 있더라도 미분계수가 0에 가까워져 결국 최소 지점에 도달하게 된다.
   </ol>

$$temp0:=\theta_0-\alpha{\partial \over\partial\theta_0}J(\theta_0,\theta_1)\\temp1:=\theta_1-\alpha{\partial \over\partial\theta_1}J(\theta_0,\theta_1)\\\theta_0 := temp0, \  \theta_1:=temp1$$

 θ0와 θ1을 동시에 할당하기 위해서는 각각 우변에서 연산한 뒤에 temp0와 temp1에 할당해주고 그 뒤에 temp0와 temp1을 θ0와 θ1에 각각 할당해야 한다.

예를 들어 temp0에 연산된 값을 할당하고 바로 θ0에 할당하게 되면 그 뒤에 연산하는 temp1에서 연산 되는 J(θ0,θ1)값이 바뀌게 되어 올바르지 않은 값이 할당될 수 있기때문에 주의 해야한다.

<strong><u>**linear regression with gradient descent **</u></strong>

j = 0 :$$\alpha{\partial \over\partial\theta_0}J(\theta_0,\theta_1) = \frac{1}{m}\sum_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})\\$$
j = 1 :$$\alpha{\partial \over\partial\theta_1}J(\theta_0,\theta_1) = \frac{1}{m}\sum_{i=1}^m(h_\theta((x)^{(i)})-y^{(i)})x^{(i)}\\$$

이 식을 gradient descent algorithm에 대입한다.   

cost function에서 가장 오른쪽에 있는 점을 기준으로 초기화했을 때 training set과 전혀 맞지 않는 직선이 만들어지기 때문에 gradient descent를 사용하여 최소 지점에 도달하려고 한다. 

 cost function에서 초기화한 점을 기준으로 그 주위의 가장 가파른 기울기를 가진 곳으로 이동하여 gradient descent 과정을 반복하면 linear regression cost function은 항상 convex function이고 하나의 최솟값만 가지기 때문에 global minimum에 도달하게 된다. 따라서 training data에 오차가 가장 적은 hθ(x)를 구하여 새로운 데이터가 생겼을 때 어떤 식의 값을 가지게 될지 유추가 가능해진다. 

=> 다음 섹션에는 gradient descent를 반복하는 과정을 생략할 수 있도록 하는 방법에 대하여 배운다.