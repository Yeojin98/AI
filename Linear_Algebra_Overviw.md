<strong><u>Matrics_overview</u></strong>

1. Dimension of a matrix are [Rows x Columns]
2. Matrix elements $A_{(i,j)}$ = entry in $i^{th}$ row and jth column.

<strong><u>Vectors_overview</u></strong>

1.  Vector elements $v_i$ = $i^{th}$ element of the vector
2. 주로 수학에서는 요소를 1부터 세는 1-indexed가 자주 쓰이지만 machine learning에서는 0-indexed가 유용하게 사용된다.

<strong><u>Matrix manipulation</u></strong>

1. Addition 

   -> Can only add matrices of the same dimensions.

   -> 덧셈의 결과값이 original matrices 같은 dimensions을 가진다.

2. Multiplication by scalar

   -> Multiply each element by the scalar

   -> 스칼라 곱의 결과값이 original matrices의 size와 같다.

3. Division by a scalar

   -> Each element is divided by the scalar

4. Matrix by vector multiplication

   -> More generally if [a x b] * [b x c] ,then new matrix is [a x c]

   -> vector multiplication은 inner dimension이 같아야 연산이 가능하다.

   A * x = y 에서 A는 [m x n], x는 [n x 1] matrix라 했을 때, 

   $y_i$는 A의 i번째 행에 x의 모든 요소를 곱하여 더한 값이다. 

   

   ex) House size: 2104, 1416, 1534, 852  $h_\theta(x) = -40+0.25x$

   prediction = $\begin{vmatrix} 1 & 2104 \\ 1 & 1416 \\ 1 & 1534 \\ 1 & 852\end{vmatrix} \times$ $\begin{vmatrix}  -40 \\ 0.25 \end{vmatrix}=  $  $\begin{vmatrix} -40\times 1+0.25\times 2104\\ -40\times 1+0.25\times 1416 \\ -40\times 1+0.25\times 1534 \\ -40\times 1+0.25\times 852\end{vmatrix} $ 

   위 식에서 새롭게 추가된 1s의 column은 house size외에 침대의 개수 등($\theta_0$) 새로운 변수가 추가되어 prediction을 계산할 수 있다는 의미이다.

   => Prediction = Data Matrix * Parameters

   이러한 계산 방식은 for문을 4번, 혹은 데이터가 늘어남에 따라 1000번을 반복하는 것보다 훨씬 효율적이며 코딩하기가 쉽다.

   

   mechanism : A x B = C, A = [mxn] B = [mxo] C= [mx0]

   -> B matrix의 하나의 열을 vector로 두고 A의 각 행과 mutiply하게 [mx1] vector가 만들어진다. 이것을 B의 열 갯수만큼 반복하면 o개의 열이 만들어져 최종 결과는 C=[mxo]가 되는 것이다.

<strong><u>Implementation/Use</u></strong>

ex) House prices에 3 competing hypotheses가 있다면?

$h_\theta(x) = -40 +0.25x$, $h_\theta(x) = 200 +0.1x$, $h_\theta(x) = -150 +0.4x$

=> Prediction = Data Matrix * Parameters

$\begin{vmatrix} 1 & 2104 \\ 1 & 1416 \\ 1 & 1534 \\ 1 & 852\end{vmatrix} \times$$\begin{vmatrix}  -40 & 200 & -150 \\ 0.25 & 0.1 &0.4 \end{vmatrix}$=$\begin{vmatrix}  486 & 410 & 692 \\ 314 & 342 & 416 \\ 344 & 353 & 464 \\ 173 & 285 & 191 \end{vmatrix}$

4개의 집 크기 데이터와 3개의 hypotheses를 한번의 연산으로 끝낼 수 있다. 

<strong><u>Matrix multiplication properties</u></strong>

연산 시 주의해야할 matrix의 특성

1. Commutativity : $\ A \times B $ != $B \times A$

2. Associativity :$\ (A \times B)\times C$ != $A \times (B \times C)$

3. Identity matrix: 대각선을 따라 있는 elements는 모두 1, 나머지는 o

   ex) $I_{\{{3\times 3\}}} = \begin{vmatrix}  1&0&0 \\ 0&1&0\\ 0&0&1 \end{vmatrix}$

   -> $A\times B$에서 B가 단위행렬이라면 AB = =BA,

   -> $A\times I$  = A,  $I\times A$ = A 

<strong><u>Inverse and transpose operations</u></strong>

1. Inverse

   -> Only matrices which are m x m have inverses

   주어진 수에 어떤 수를 곱하여 1이 나오는 수를 Inverse 즉, 역수라고 하며 $A^{-1}$로 표기한다. 역행렬을 직접 구하는 것이 매우 어려우므로 open source library를 사용하는 것이 효과적이다.

2. matrix transpose

   -> 전치행렬은 행과 열의 위치를 바꿔서 구할 수 있다.

   ex) dimension이 [mxn]인 행렬 A가 존재한다고 가정하였을 때, 

   행렬 A의 전치행렬인 B를 구하자.

   A의 1행은 B의 1열이 되고 A의 행이 m개 이므로 m번 반복한다. 그러면 B, 즉 $A^T$의 dimension은 [nxm]이다. 

   A=  $\begin{vmatrix}  -2 & 3 & 15 \\ 6 & 32 &8 \end{vmatrix}$ $A^T$=  $\begin{vmatrix}  -2 & 6 \\ 3 & 32 \\ 15&8 \end{vmatrix}$