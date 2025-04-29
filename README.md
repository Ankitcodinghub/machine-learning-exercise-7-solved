# machine-learning-exercise-7-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Exercise 7 Solved](https://www.ankitcodinghub.com/product/machine-learning-labs-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110199&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning  Exercise 7 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
. The goal of this exercise is to

• Implement and debug Support Vector Machine (SVM) using SGD and coordinate descent.

• Derive updates for the coordinate descent algorithm for the dual optimization problem for SVM.

• Implement and debug the coordinate descent algorithm.

• Compare it to the primal solution.

Setup, data and sample code. Obtain the folder labs/ex07 of the course github repository

github.com/epfml/ML course

We will finally depart from using the height-weight dataset and instead use a toy dataset from scikit-learn. We have provided sample code templates that already contain useful snippets of code required for this exercise.

1 Support Vector Machines using SGD

Until now we have implemented linear and logistic regression to do classification. In this exercise we will use the Support Vector Machine (SVM) for classification. As we have seen in the lecture notes, the original optimization problem for the Support Vector Machine (SVM) is given by

N

λ 2

min w (1)

w

where ` : R → R, `(z) := max{0,1 − z} is the hinge loss function. Here for any n, 1 ≤ n ≤ N, the vector xn ∈ RD is the nth data example, and yn ∈ {±1} is the corresponding label.

Problem 1 (SGD for SVM):

Implement stochastic gradient descent (SGD) for the original SVM formulation (1). That is in every iteration, pick one data example n ∈ [N] uniformly at random, and perform an update on w based on the (sub-)gradient of the nth summand of the objective (1). Then iterate by picking the next n.

1. Fill in the notebook functions calculate accuracy(y, X, w) which computes the accuracy on the training/test dataset for any w and calculate primal objective(y, X, w, lambda ) which computes the total primal objective (1).

2. Derive the SGD updates for the original SVM formulation and fill in the notebook function calculate stochastic gradient() which should return the stochastic gradient of the total cost function (loss plus regularizer) with respect to w. Finally, use sgd for svm demo() provided in the template for training.

2 Support Vector Machines using Coordinate Descent

As seen in class, another approach to train SVMs is by considering the dual optimization problem given by

YXX&gt;Yα such that 0 ≤ αn ≤ 1 ∀n ∈ [N] (2)

where 1 is the vector of size N filled with ones, Y := diag(y), and X ∈ RN×D again collects all N data examples as its rows, as usual. In this approach we optimize over the dual variables α and map the solutions back to the primal vector w via the mapping we have seen in class: w(α) = λ1XTYα.

Problem 2 (Coordinate Descent for SVM):

In this part we will derive the coordinate descent (or rather ascent in our case) algorithm seen in class to solve the dual (2) of the SVM formulation. That is, at every iteration, we pick a coordinate n ∈ [N] uniformly at random, and fully optimize the objective (2) with respect to that coordinate alone. Hence one step of coordinate ascent corresponds to solving, for a given coordinate and our current vector α ∈ [0,1]N, the one dimensional problem:

such that 0 ≤ αn + γ ≤ 1 (3)

where YXX&gt;Yα and en = [0,··· ,1,··· ,0]&gt; (all zero vector except at the nth position). We denote by γ∗ the value of γ which maximises problem 3. The coordinate update is then αnew = α + γ∗en.

1. Solve problem 3 and give a closed form solution for the update αnew = α + γ∗en, this update should only involve α, λ, xn, yn and w(α) (Hint: Notice that γ →7 f(α + γen) is polynomial and don’t forget the constraints !). Convince yourself that this update can be computed in O(D) time.

2. Find an efficient way of updating the corresponding w(αnew). This should be computable in O(D) time.

3. Fill in the notebook functions calculate coordinate update() which should compute the coordinate update for a single desired coordinate and calculate dual objective() which should return the objective (loss) for the dual problem (2) .

4. Finally train your model using coordinate descent (here ascent) using the given function sgd for svm demo() in the template. Compare to your SGD implementation. Which one is faster? (Compare the training objective values (1) for the w iterates you obtain from each method). Is the gap going to 0?

Theory Excercises

Problem 3 (Kernels):

In class we have seen that many kernel functions k(x,x0) can be written as inner products φ(x)&gt;φ(x0), for a suitably chosen vector-function φ(·) (often called a feature map). Let us say that such a kernel function is valid. We further discussed many operations on valid kernel functions that result again in valid kernel functions. Here are two more.

1. Let k1(x,x0) be a valid kernel function. Let f be a polynomial with positive coefficients. Show that k(x,x0) = f(k1(x,x0)) is a valid kernel.

2. Show that k(x,x0) = exp(k1(x,x0)) is a valid kernel assuming that k1(x,x0) is a valid kernel. Hint: You can use the following property: if (Kn)n≥0 is a sequence of valid kernels and if there exists a function

K : X × X → R such that for all , then K is a valid kernel.

2
