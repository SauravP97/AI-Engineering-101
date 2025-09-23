# Deep Research Report

## Table of Contents 
- Define the formal model for a symmetric first-price sealed-bid auction with two bidders. This should include defining the bidders' private values (v), the common cumulative distribution function (F(v)) and its probability density function (f(v)), the concept of a symmetric bidding strategy B(v), and the rules of the auction.
- Formulate a single bidder's expected profit (utility) function. This formulation must express the bidder's profit in terms of their private value (v) and their bid (b), incorporating the probability of the other bidder placing a lower bid based on the symmetric strategy B(v).
- Derive the symmetric Bayesian Nash Equilibrium bidding strategy, B(v), by maximizing the expected profit function with respect to the bidder's own bid. This involves taking the first-order condition, solving the resulting differential equation, and specifying the final bidding function in terms of the value distribution F(v).
- "Analyze the equilibrium bidding strategy in a standard symmetric first-price sealed-bid auction. Detail the derivation of the bidding function, assuming bidders' private values are drawn from the same distribution F(v). This will serve as a baseline for the asymmetric case.",
- "Set up the theoretical framework for an asymmetric first-price sealed-bid auction with two bidders. Define the assumptions, the different value distributions (F1(v) and F2(v)), and formulate the expected payoff function for each bidder. Derive the first-order conditions for the Bayesian Nash Equilibrium.",
- "From the first-order conditions of the asymmetric model, derive the system of coupled differential equations that characterizes the inverse equilibrium bidding functions. Specify the necessary boundary conditions required to solve this system."
- Is there a general closed-form solution to the system of differential equations that describe the equilibrium in the asymmetric first-price sealed-bid auction? Investigate the conditions under which a general analytical solution exists or does not exist.
- Detail the solution and derivation of the equilibrium bidding functions in asymmetric first-price auctions where bidders' valuations are drawn from uniform distributions with different upper bounds (e.g., U[0, b1] and U[0, b2]).
- Describe the equilibrium bidding strategies for asymmetric first-price auctions when bidders' private values are drawn from exponential distributions with different parameters (e.g., different mean values).
- Identify and summarize other well-known, specific examples of asymmetric first-price auctions with explicitly solved equilibrium bidding functions, beyond the standard uniform and exponential distribution cases. This could include cases with power-law distributions or other named distributions.
- In cases where an analytical or closed-form solution cannot be found, what are the primary numerical and computational methods used to approximate the equilibrium bidding strategies for asymmetric first-price auctions?

## Report 
## Is there a general method for solving a first-price sealed-bid auction with two bidders who have independent private values drawn from different distributions (i.e., ex-ante asymmetric bidders)?



## What is the standard methodology for solving a first-price sealed-bid auction when the two bidders' private values are drawn from the same distribution (i.e., the symmetric case)? This should include the derivation of the Bayesian Nash Equilibrium bidding strategy.



 
 ### Define the formal model for a symmetric first-price sealed-bid auction with two bidders. This should include defining the bidders' private values (v), the common cumulative distribution function (F(v)) and its probability density function (f(v)), the concept of a symmetric bidding strategy B(v), and the rules of the auction.

### Formal Model: Symmetric First-Price Sealed-Bid Auction with Two Bidders

The formal model for a symmetric first-price sealed-bid auction with two bidders is a game of incomplete information, defined by the following components:

**1. Bidders and Private Values (v)**

*   There are two bidders, indexed by i = 1, 2.
*   Each bidder `i` has a **private value** `vi` for the object being auctioned. This value represents the maximum amount the bidder is willing to pay.
*   These values are private information; a bidder knows their own value but not the value of the other bidder.
*   The bidders' values are treated as independent and identically distributed (i.i.d.) random variables drawn from a common probability distribution. This means each bidder's value is drawn from the same "pool" of potential values without influencing the other's draw (cs.brown.edu, comlabgames.com). For example, values might be drawn from the interval [0,1] (chegg.com).

**2. Common Distribution and Density Functions (F(v) and f(v))**

*   The bidders' private values are drawn from a common **cumulative distribution function (CDF)**, denoted as `F(v)`. `F(v)` represents the probability that a bidder's private value is less than or equal to `v`. So, `F(v) = P(vi ≤ v)`.
*   The associated **probability density function (PDF)** is denoted as `f(v)`, where `f(v) = F'(v)`. This function describes the relative likelihood for a random variable to take on a given value.

**3. Symmetric Bidding Strategy B(v)**

*   In a symmetric auction, it is assumed that both bidders employ the same bidding strategy. This strategy is a function, `B(v)`, which maps a bidder's private value `v` to a specific bid `b`.
*   Therefore, if a bidder `i` has a private value `vi`, they will submit a bid `bi = B(vi)`.
*   This strategy `B(v)` is the solution in a symmetric Bayes-Nash equilibrium. It is a function where a bidder maximizes their expected payoff given their own value and the fact that the other bidder is also using the same strategy function. For a two-bidder (`n=2`) auction, the equilibrium bidding strategy is given by the formula: `b(vi) = vi - ∫[0 to vi] F(x) dx / F(vi)` (cs.brown.edu).

**4. Rules of the Auction**

*   **Sealed Bids:** Each bidder `i` secretly submits a non-negative bid, `bi` (econgraphs.org).
*   **Winner Determination:** The bidder who submits the highest bid wins the object.
*   **Payment:** The winner pays the amount of their own bid.
*   **Payoffs:**
    *   The winning bidder's payoff is their private value minus their bid: `vi - bi`.
    *   The losing bidder's payoff is zero.
*   **Ties:** In the event of a tie (where `b1 = b2`), a tie-breaking rule is applied, commonly assigning the object to one bidder with equal probability (e.g., a coin flip).

 
 ### Formulate a single bidder's expected profit (utility) function. This formulation must express the bidder's profit in terms of their private value (v) and their bid (b), incorporating the probability of the other bidder placing a lower bid based on the symmetric strategy B(v).

A single bidder's expected profit (utility) function in a sealed-bid auction with a symmetric bidding strategy can be formulated as follows:

**π(v, b) = (v - b) * F(B⁻¹(b))**

Where:
*   **π(v, b)** represents the expected profit for a bidder with a private value `v` who places a bid `b`.
*   **v** is the bidder's private value, which is the maximum amount they are willing to pay for the item.
*   **b** is the bid the bidder places.
*   **(v - b)** is the bidder's profit or surplus if they win the auction.
*   **B(v)** is the symmetric bidding strategy function. It is assumed that all bidders use the same strategy, where their bid is a function of their private value. This function is strictly increasing, meaning that a higher private value results in a higher bid.
*   **B⁻¹(b)** is the inverse of the bidding strategy. It determines the private value `v` that corresponds to a given bid `b`.
*   **F(v)** is the cumulative distribution function (CDF) for the bidders' private values. `F(x)` gives the probability that a bidder's private value is less than or equal to `x`.

### Breakdown of the Formulation:

A bidder's expected profit is calculated by multiplying the potential profit from winning by the probability of winning.

1.  **Profit from Winning**: If the bidder wins the auction, their profit is the difference between their private value for the item (`v`) and the amount they bid (`b`). This is represented by the term `(v - b)`. If they lose, their profit is zero.

2.  **Probability of Winning**: The bidder wins if their bid `b` is higher than the other bidder's bid. Since the other bidder is using the same symmetric strategy `B(v)`, their bid will be `B(v')`, where `v'` is the other bidder's private value.

    *   Therefore, to win, `b` must be greater than `B(v')`.
    *   Because the bidding function `B(v)` is strictly increasing, we can use its inverse `B⁻¹(b)` to find the equivalent condition for the other bidder's value: `B⁻¹(b) > v'`.
    *   The probability that the other bidder's value `v'` is less than `B⁻¹(b)` is given by the cumulative distribution function `F(B⁻¹(b))`. This term, `F(B⁻¹(b))`, therefore represents the probability of the first bidder winning the auction with a bid of `b`.

By combining these two components, we arrive at the comprehensive formula for a bidder's expected profit. This function is fundamental in auction theory for analyzing and determining optimal bidding strategies.

 
 ### Derive the symmetric Bayesian Nash Equilibrium bidding strategy, B(v), by maximizing the expected profit function with respect to the bidder's own bid. This involves taking the first-order condition, solving the resulting differential equation, and specifying the final bidding function in terms of the value distribution F(v).

### Derivation of the Symmetric Bayesian Nash Equilibrium Bidding Strategy

In a first-price, sealed-bid auction with `n` risk-neutral bidders, where each bidder's private value `v` is independently and identically drawn from a cumulative distribution function `F(v)` with a corresponding probability density function `f(v)` over a support (commonly `[0, V]`), we seek to find a symmetric Bayesian Nash Equilibrium (BNE) bidding strategy, `B(v)`. This strategy is symmetric because all bidders are assumed to use the same function `B(.)` to determine their bid based on their private value.

#### 1. The Bidder's Expected Profit Function

A bidder `i` with a private value `v` who submits a bid `b` earns a profit (or utility) of `v - b` if they win the auction and `0` if they lose. The bidder's objective is to choose a bid `b` that maximizes their expected profit.

The expected profit `E[π(b, v)]` is the product of the profit from winning and the probability of winning:

`E[π(b, v)] = (v - b) * Pr(b is the highest bid)`

Assuming that all other `n-1` bidders follow the strictly increasing and differentiable strategy `B(v)`, bidder `i` wins if their bid `b` is greater than all other bidders' bids.

`Pr(Win with bid b) = Pr(b > B(v_j) for all j ≠ i)`

Since `B(v)` is strictly increasing, it has a well-defined inverse, `B⁻¹(b)`. Therefore, the condition `b > B(v_j)` is equivalent to `B⁻¹(b) > v_j`. The probability that any single opponent's value `v_j` is less than `B⁻¹(b)` is given by the cumulative distribution `F(B⁻¹(b))`.

Because all bidders' values are drawn independently, the probability that all `n-1` opponents have values less than `B⁻¹(b)` is:

`Pr(Win with bid b) = [F(B⁻¹(b))]^(n-1)`

Thus, the expected profit function for a bidder with value `v` who bids `b` is:

`E[π(b, v)] = (v - b) * [F(B⁻¹(b))]^(n-1)`

#### 2. Maximization and the First-Order Condition

To find the bid `b` that maximizes this expected profit, we take the derivative of the expected profit function with respect to `b` and set it to zero. This is the first-order condition (FOC) for a maximum. Using the product rule for differentiation:

`d/db [E[π(b, v)]] = -[F(B⁻¹(b))]^(n-1) + (v - b) * (n-1)[F(B⁻¹(b))]^(n-2) * f(B⁻¹(b)) * (1 / B'(B⁻¹(b))) = 0`

#### 3. Imposing the Symmetric Equilibrium Condition

In a symmetric BNE, the optimal bid for a player with value `v` must be `b = B(v)`. By substituting `b = B(v)` into the FOC, we also have `B⁻¹(b) = v`. This simplifies the FOC significantly:

`-[F(v)]^(n-1) + (v - B(v)) * (n-1)[F(v)]^(n-2) * f(v) * (1 / B'(v)) = 0`

This equation represents a first-order ordinary differential equation (ODE) that defines the equilibrium bidding strategy `B(v)`.

#### 4. Solving the Differential Equation

We can rearrange the ODE to solve for `B(v)` (https://hanzhezhang.github.io/teaching/Chicago_ECON207/207sol_auction.pdf).

`B'(v) * [F(v)]^(n-1) = (v - B(v)) * (n-1)[F(v)]^(n-2) * f(v)`

Rearranging terms to group `B(v)` and `B'(v)`:

`B'(v)[F(v)]^(n-1) + B(v)(n-1)[F(v)]^(n-2)f(v) = v(n-1)[F(v)]^(n-2)f(v)`

The left side of the equation is the derivative of the product `B(v)[F(v)]^(n-1)` with respect to `v`.

`d/dv [B(v)[F(v)]^(n-1)] = v(n-1)[F(v)]^(n-2)f(v)`

We can now integrate both sides from the lowest possible value (assumed to be 0) up to `v`:

`∫[0 to v] d/dx [B(x)[F(x)]^(n-1)] dx = ∫[0 to v] x(n-1)[F(x)]^(n-2)f(x) dx`

`B(v)[F(v)]^(n-1) - B(0)[F(0)]^(n-1) = ∫[0 to v] x(n-1)[F(x)]^(n-2)f(x) dx`

A crucial boundary condition is that a bidder with a value of 0 will bid 0, so `B(0) = 0`. This simplifies the equation to:

`B(v)[F(v)]^(n-1) = ∫[0 to v] x * d/dx([F(x)]^(n-1)) dx`

We can solve the integral on the right-hand side using integration by parts, which yields:

`∫[0 to v] x * d/dx([F(x)]^(n-1)) dx = v[F(v)]^(n-1) - ∫[0 to v] [F(x)]^(n-1) dx`

Substituting this back gives:

`B(v)[F(v)]^(n-1) = v[F(v)]^(n-1) - ∫[0 to v] [F(x)]^(n-1) dx`

#### 5. The Final Bidding Function

Finally, dividing by `[F(v)]^(n-1)` gives the explicit symmetric Bayesian Nash Equilibrium bidding strategy `B(v)`:

`B(v) = v - (∫[0 to v] [F(x)]^(n-1) dx) / [F(v)]^(n-1)`

This can be expressed more compactly as:

`B(v) = v - ∫[0 to v] (F(x) / F(v))^(n-1) dx`

This function shows that a bidder's optimal strategy is to bid their true value `v` minus a "shade" or discount. This discount term, represented by the integral, accounts for the trade-off between increasing the bid to improve the probability of winning and decreasing the bid to increase the profit if one does win. The size of this shade depends on the bidder's own value `v` and the distribution of values `F(v)` for the other `n-1` bidders.

## How does the analysis change when the bidders are asymmetric, i.e., their private values are drawn from different distributions? Detail the system of differential equations and boundary conditions that characterize the equilibrium bidding strategies in this asymmetric case.



 
 ### "Analyze the equilibrium bidding strategy in a standard symmetric first-price sealed-bid auction. Detail the derivation of the bidding function, assuming bidders' private values are drawn from the same distribution F(v). This will serve as a baseline for the asymmetric case.",

### Equilibrium Bidding Strategy in a Symmetric First-Price Sealed-Bid Auction

In a standard symmetric first-price sealed-bid auction, the equilibrium bidding strategy is a foundational concept in auction theory. This analysis details the model's assumptions, the derivation of the bidding function, and its interpretation, which serves as a crucial baseline for understanding more complex scenarios, such as auctions with asymmetric bidders.

#### **1. Model Assumptions**

The standard model for a symmetric first-price sealed-bid auction is built on the following assumptions:

*   **Number of Bidders:** There are *n* bidders competing to acquire a single item.
*   **Private Values:** Each bidder *i* has a private value *v<sub>i</sub>* for the item. This value represents the maximum amount they are willing to pay.
*   **Value Distribution:** The bidders' values are independent and identically distributed (i.i.d.), drawn from a common, strictly increasing, and continuous distribution function *F(v)*, with a corresponding probability density function *f(v)* over a support, typically normalized to [0, V].
*   **Risk Neutrality:** Bidders are assumed to be risk-neutral, meaning they aim to maximize their expected payoff. The payoff for winning is the bidder's private value minus their bid (*v<sub>i</sub>* - *b<sub>i</sub>*), and the payoff for losing is zero.
*   **Symmetric Equilibrium:** We look for a symmetric Bayesian-Nash equilibrium, where all bidders employ the same bidding strategy, denoted by a function *b(v)*. This function is assumed to be strictly increasing and differentiable.

#### **2. The Bidder's Optimization Problem**

A bidder with value *v* must choose a bid *b* to maximize their expected payoff. The expected payoff, *U(b, v)*, is the product of the probability of winning and the surplus gained if they win.

*   **Surplus:** If the bidder wins, their surplus is *v - b*.
*   **Probability of Winning:** A bidder wins if their bid *b* is higher than all other *n-1* bids. Since all bidders are assumed to use the same increasing strategy *b(v)*, a bidder with value *v* bidding *b(v)* will win if their value *v* is the highest among all *n* bidders. The probability that any single other bidder *j* has a value less than *v* is *F(v)*. Given the i.i.d. assumption, the probability that all *n-1* other bidders have values less than *v* is *F(v)<sup>n-1</sup>*.

Therefore, the expected payoff for a bidder with value *v* who bids according to the strategy *b(v)* is:

*U(v) = (v - b(v)) * F(v)<sup>n-1</sup>*

#### **3. Derivation of the Equilibrium Bidding Function**

To find the equilibrium bidding function, we use the first-order condition. In a Bayesian-Nash equilibrium, no bidder can improve their expected payoff by unilaterally changing their strategy. This means that for a bidder with true value *v*, bidding *b(v)* must be optimal.

Consider a bidder with value *v* contemplating a bid *b(z)*, effectively pretending their value is *z*. Their expected utility is:

*U(z, v) = (v - b(z)) * F(z)<sup>n-1</sup>*

To maximize this with respect to their choice of *z*, we take the derivative with respect to *z* and set it to zero:

*∂U/∂z = -b'(z)F(z)<sup>n-1</sup> + (v - b(z))(n-1)F(z)<sup>n-2</sup>f(z) = 0*

The principle of incentive compatibility dictates that in equilibrium, each bidder will truthfully reveal their type, meaning they will choose *z = v*. Substituting *v = z* into the first-order condition gives us a differential equation that defines the equilibrium bidding function *b(v)*:

*-b'(v)F(v)<sup>n-1</sup> + (v - b(v))(n-1)F(v)<sup>n-2</sup>f(v) = 0*

Rearranging this equation yields:

*b'(v)F(v)<sup>n-1</sup> = (v - b(v))(n-1)F(v)<sup>n-2</sup>f(v)*

This is a first-order linear differential equation. We can solve it by noticing that the term *(n-1)F(v)<sup>n-2</sup>f(v)* is the derivative of *F(v)<sup>n-1</sup>*. Let's define a function *G(v) = b(v)F(v)<sup>n-1</sup>*. Its derivative with respect to *v* is:

*G'(v) = b'(v)F(v)<sup>n-1</sup> + b(v)(n-1)F(v)<sup>n-2</sup>f(v)*

By substituting the result from our first-order condition into this expression, we get:

*G'(v) = (v - b(v))(n-1)F(v)<sup>n-2</sup>f(v) + b(v)(n-1)F(v)<sup>n-2</sup>f(v)*
*G'(v) = v(n-1)F(v)<sup>n-2</sup>f(v)*

Now, we integrate *G'(v)* from the lowest possible value (0) up to *v*:

*∫<sub>0</sub><sup>v</sup> G'(x) dx = ∫<sub>0</sub><sup>v</sup> x(n-1)F(x)<sup>n-2</sup>f(x) dx*

This gives:

*G(v) - G(0) = ∫<sub>0</sub><sup>v</sup> x(n-1)F(x)<sup>n-2</sup>f(x) dx*

Since a bidder with a value of 0 will bid 0 (*b(0) = 0*), we have *G(0) = 0*. Substituting back *G(v) = b(v)F(v)<sup>n-1</sup>*:

*b(v)F(v)<sup>n-1</sup> = ∫<sub>0</sub><sup>v</sup> x(n-1)F(x)<sup>n-2</sup>f(x) dx*

Solving for *b(v)*, we arrive at the equilibrium bidding function:

**b(v) = [∫<sub>0</sub><sup>v</sup> x(n-1)F(x)<sup>n-2</sup>f(x) dx] / F(v)<sup>n-1</sup>**

This can be simplified using integration by parts, leading to an alternative and more intuitive form (cs.brown.edu, 2020):

**b(v) = v - [∫<sub>0</sub><sup>v</sup> F(x)<sup>n-1</sup> dx] / F(v)<sup>n-1</sup>**

#### **4. Interpretation of the Bidding Function**

The derived formula reveals that the optimal strategy is not to bid one's true value. Instead, a bidder "shades" their bid downwards.

*   **b(v) = E[Y<sub>1</sub> | Y<sub>1</sub> < v]**

The term *b(v)* represents the expected value of the highest of the other *n-1* bidders' valuations, given that one's own value *v* is the highest. This is the core insight: to win, a bidder only needs to bid slightly more than the second-highest bid. The optimal strategy, therefore, is to bid the expected value of the second-highest valuation, conditional on your own value being the winning one (diva-portal.org, n.d.).

The second form of the equation, *b(v) = v - [markdown]*, clearly shows that the bid is the true value *v* minus a markdown. This markdown term decreases as the number of bidders (*n*) increases. With more competitors, the highest of the other bids is likely to be higher, forcing each bidder to bid more aggressively (i.e., closer to their true value).

#### **5. Example: Uniform Distribution**

To illustrate, assume bidders' values are drawn from a uniform distribution on [0, 1]. In this case, *F(v) = v* and *f(v) = 1*.

Substituting into the bidding function:

*b(v) = v - [∫<sub>0</sub><sup>v</sup> x<sup>n-1</sup> dx] / v<sup>n-1</sup>*
*b(v) = v - [(v<sup>n</sup>/n)] / v<sup>n-1</sup>*
*b(v) = v - v/n*
*b(v) = (n-1)/n * v*

Thus, in a first-price auction with values drawn from a uniform distribution, the symmetric Bayesian-Nash equilibrium strategy is for each bidder to bid a fraction *(n-1)/n* of their true value (homepages.math.uic.edu, n.d.). For example, with two bidders, the strategy is to bid half of one's value. With ten bidders, it's to bid 90% of one's value.

This detailed analysis of the symmetric case provides the necessary foundation for examining asymmetric auctions, where bidders draw their values from different distributions, leading to more complex strategic interactions.

**References**
*   cs.brown.edu. (2020). *First-Price, Sealed-Bid Auctions*. Retrieved from https://cs.brown.edu/courses/cs1951k/lectures/2020/first_price_auctions.pdf
*   diva-portal.org. (n.d.). *A Comparison of the First-Price and Second-Price Auctions*. Retrieved from http://www.diva-portal.org/smash/get/diva2:1245417/FULLTEXT01.pdf
*   homepages.math.uic.edu. (n.d.). *Auctions*. Retrieved from https://homepages.math.uic.edu/~marker/stat473-F14/auctions.pdf

 
 ### "Set up the theoretical framework for an asymmetric first-price sealed-bid auction with two bidders. Define the assumptions, the different value distributions (F1(v) and F2(v)), and formulate the expected payoff function for each bidder. Derive the first-order conditions for the Bayesian Nash Equilibrium.",

### Theoretical Framework for an Asymmetric First-Price Sealed-Bid Auction

This report outlines the theoretical framework for a first-price sealed-bid auction with two asymmetric bidders. It will define the model's assumptions, the bidders' value distributions, formulate the expected payoff functions, and derive the first-order conditions that characterize the Bayesian Nash Equilibrium.

#### 1. Assumptions and Model Setup

The foundation of the asymmetric first-price sealed-bid auction model rests on the following assumptions:

*   **Two Bidders:** The auction involves two bidders, indexed as bidder 1 and bidder 2.
*   **Private, Independent Values:** Each bidder `i` has a private value `v_i` for the single item being auctioned. This value represents the maximum amount the bidder is willing to pay. The values are known only to the bidders themselves.
*   **Asymmetric Value Distributions:** The bidders' private values are drawn independently from different probability distributions.
    *   Bidder 1's value, `v_1`, is drawn from a distribution `F_1(v)` with a corresponding probability density function `f_1(v)`.
    *   Bidder 2's value, `v_2`, is drawn from a distribution `F_2(v)` with a corresponding probability density function `f_2(v)`.
    *   Both distributions are assumed to have a common support, typically normalized to `[0, 1]`. The functions `F_1` and `F_2` are continuous and strictly increasing.
*   **Risk Neutrality:** Bidders are risk-neutral, meaning their objective is to maximize their expected payoff. The payoff for a winning bidder `i` is their value minus their bid (`v_i - b_i`), and the payoff for a losing bidder is zero.
*   **Sealed Bids:** Bidders submit their bids simultaneously in a sealed fashion. The highest bidder wins the auction.
*   **First-Price Rule:** The winning bidder pays the amount of their own bid.
*   **Bayesian Game:** The structure of the game, including the bidders' value distributions (`F_1` and `F_2`), is common knowledge. Bidders know their own value but not the value of their opponent.

#### 2. Expected Payoff Function

In this Bayesian game, a bidder's strategy is a function that maps their private value to a bid. Let `b_1(v_1)` and `b_2(v_2)` be the bidding strategies for bidder 1 and bidder 2, respectively. A key feature of a Bayesian Nash Equilibrium in this setting is that these bidding strategies are strictly increasing and differentiable.

Let's formulate the expected payoff for Bidder 1.

For Bidder 1, with a private value `v_1`, who submits a bid `b_1`, the payoff is `(v_1 - b_1)` if they win, and 0 if they lose. Bidder 1 wins if their bid is higher than Bidder 2's bid, i.e., `b_1 > b_2(v_2)`.

The probability of winning for Bidder 1 with a bid `b_1` is the probability that Bidder 2's bid is less than `b_1`:
`P(Win | b_1) = P(b_2(v_2) < b_1)`

Since the bidding strategy `b_2(v_2)` is strictly increasing, it has a well-defined inverse function, `b_2^{-1}(b)`. Therefore, we can express the winning condition in terms of Bidder 2's value:
`P(Win | b_1) = P(v_2 < b_2^{-1}(b_1))`

Given that `v_2` is drawn from the distribution `F_2`, this probability is:
`P(Win | b_1) = F_2(b_2^{-1}(b_1))`

The **expected payoff for Bidder 1** with value `v_1` and bid `b_1` is the product of the payoff if they win and the probability of winning:
`U_1(b_1, v_1) = (v_1 - b_1) * F_2(b_2^{-1}(b_1))`

Symmetrically, the **expected payoff for Bidder 2** with value `v_2` and bid `b_2` is:
`U_2(b_2, v_2) = (v_2 - b_2) * F_1(b_1^{-1}(b_2))`

#### 3. Derivation of First-Order Conditions for Bayesian Nash Equilibrium

A Bayesian Nash Equilibrium (BNE) is a set of strategies, `(b_1^*(v_1), b_2^*(v_2))`, such that each bidder's strategy maximizes their expected payoff, given the other bidder's strategy. To find these equilibrium strategies, we use optimization. Each bidder chooses their bid `b_i` to maximize their expected payoff `U_i`. The first-order conditions are found by taking the derivative of the expected payoff function with respect to the bid and setting it to zero.

**For Bidder 1:**
We differentiate `U_1(b_1, v_1)` with respect to `b_1`:
`∂U_1 / ∂b_1 = -F_2(b_2^{-1}(b_1)) + (v_1 - b_1) * d/db_1 [F_2(b_2^{-1}(b_1))]`

Using the chain rule for the second term:
`d/db_1 [F_2(b_2^{-1}(b_1))] = f_2(b_2^{-1}(b_1)) * (b_2^{-1})'(b_1)`

Setting the derivative to zero gives the first-order condition for Bidder 1's optimal bid:
`-F_2(b_2^{-1}(b_1)) + (v_1 - b_1) * f_2(b_2^{-1}(b_1)) * (b_2^{-1})'(b_1) = 0`

**For Bidder 2:**
Similarly, we differentiate `U_2(b_2, v_2)` with respect to `b_2` and set it to zero to get the first-order condition for Bidder 2:
`-F_1(b_1^{-1}(b_2)) + (v_2 - b_2) * f_1(b_1^{-1}(b_2)) * (b_1^{-1})'(b_2) = 0`

These two equations form a system of coupled differential equations. In equilibrium, the bid `b_1` must be the result of Bidder 1's strategy, `b_1(v_1)`, and `b_2` must be `b_2(v_2)`. The solution to this system, along with appropriate boundary conditions (typically that a bidder with a value of 0 will bid 0), defines the pair of equilibrium bidding strategies `(b_1^*(v_1), b_2^*(v_2))` for the asymmetric first-price auction.

While a general closed-form solution for any pair of distributions `F_1` and `F_2` is not typically available, this framework provides the necessary conditions that any Bayesian Nash Equilibrium must satisfy. Specific solutions can be found for particular distributions, such as uniform distributions over different supports (Griesmer, et al., 1967) (Hafalir-Vijay.pdf).

 
 ### "From the first-order conditions of the asymmetric model, derive the system of coupled differential equations that characterizes the inverse equilibrium bidding functions. Specify the necessary boundary conditions required to solve this system."

### Derivation of Coupled Differential Equations for Inverse Bidding Functions

In an asymmetric first-price sealed-bid auction with two risk-neutral bidders (i=1, 2), the private values, `v_i`, are drawn independently from different cumulative distribution functions, `F_i`, with continuous probability density functions, `f_i`, over a common support (for simplicity, let's assume `[0, w]`).

An equilibrium is characterized by a pair of strictly increasing and differentiable bidding strategies, `b_i(v_i)`. Let `φ_i(b) = v_i` be the inverse bidding function, which denotes the value of a bidder `i` who submits a bid `b`.

1.  **Expected Payoff:**
    The expected payoff for bidder 1, with a value `v_1`, who submits a bid `b`, is the product of the potential gain `(v_1 - b)` and the probability of winning. The probability of winning is the probability that the other bidder's bid, `b_2(v_2)`, is less than `b`.

    *   `π_1(v_1, b) = (v_1 - b) * Pr(b_2(v_2) < b)`

    Since `b_2` is strictly increasing, `b_2(v_2) < b` is equivalent to `v_2 < φ_2(b)`. Therefore, the probability of winning is `F_2(φ_2(b))`.

    The expected payoffs are:
    *   For bidder 1: `π_1(v_1, b) = (v_1 - b) * F_2(φ_2(b))`
    *   For bidder 2: `π_2(v_2, b) = (v_2 - b) * F_1(φ_1(b))`

2.  **First-Order Conditions:**
    In equilibrium, each bidder chooses a bid `b` to maximize their expected payoff. To find this maximum, we take the derivative of the payoff function with respect to `b` and set it to zero.

    For bidder 1, the first-order condition (FOC) is:
    `∂π_1/∂b = -F_2(φ_2(b)) + (v_1 - b) * f_2(φ_2(b)) * φ_2'(b) = 0`

    In equilibrium, the optimal bid for a player with value `v_1` is `b = b_1(v_1)`, which implies `v_1 = φ_1(b)`. Substituting this into the FOC gives:
    `-F_2(φ_2(b)) + (φ_1(b) - b) * f_2(φ_2(b)) * φ_2'(b) = 0`

    Similarly, for bidder 2, the FOC is:
    `∂π_2/∂b = -F_1(φ_1(b)) + (v_2 - b) * f_1(φ_1(b)) * φ_1'(b) = 0`

    Substituting `v_2 = φ_2(b)` gives:
    `-F_1(φ_1(b)) + (φ_2(b) - b) * f_1(φ_1(b)) * φ_1'(b) = 0`

3.  **System of Coupled Differential Equations:**
    By rearranging the two equilibrium conditions from the FOCs, we can isolate the derivatives `φ_1'(b)` and `φ_2'(b)`. This yields a system of two coupled first-order ordinary differential equations that characterize the inverse equilibrium bidding functions.

    From bidder 2's FOC, we solve for `φ_1'(b)`:
    `(φ_2(b) - b) * f_1(φ_1(b)) * φ_1'(b) = F_1(φ_1(b))`
    **`φ_1'(b) = F_1(φ_1(b)) / [(φ_2(b) - b) * f_1(φ_1(b))]`**

    From bidder 1's FOC, we solve for `φ_2'(b)`:
    `(φ_1(b) - b) * f_2(φ_2(b)) * φ_2'(b) = F_2(φ_2(b))`
    **`φ_2'(b) = F_2(φ_2(b)) / [(φ_1(b) - b) * f_2(φ_2(b))]`**

    This pair of equations forms the system of coupled differential equations that implicitly defines the equilibrium bidding behavior.

### Necessary Boundary Conditions

To solve this system of differential equations, we need boundary conditions to determine a unique solution from the family of possible solutions.

The necessary boundary conditions are established by considering the behavior of bidders with the lowest possible valuation. In this model, the lowest possible value is 0 for both bidders. A bidder with a value of 0 will never bid a positive amount, as this would guarantee a non-positive payoff (zero if they lose, negative if they win). Therefore, a bidder with value 0 submits a bid of 0.

*   `b_1(0) = 0`
*   `b_2(0) = 0`

Translating these into the inverse bidding functions, `φ_i(b)`, we get the initial conditions for the system at `b=0`:

*   **`φ_1(0) = 0`**
*   **`φ_2(0) = 0`**

These two conditions are required to "pin down" the solution to the system. It is important to note that the system of differential equations is singular at this point (i.e., the denominators become zero), which means that direct numerical integration starting from `b=0` is not feasible. Specialized numerical methods or analytical techniques, such as power series expansions around the origin, are typically required to initiate a solution.

 
 ### Is there a general closed-form solution to the system of differential equations that describe the equilibrium in the asymmetric first-price sealed-bid auction? Investigate the conditions under which a general analytical solution exists or does not exist.

Based on the provided information, there is **no general closed-form solution** to the system of differential equations that describes the equilibrium in all asymmetric first-price sealed-bid auctions. The existence of an analytical solution is conditional on specific, often simplified, assumptions about the auction's structure.

### Conditions for the Existence of an Analytical Solution:

A closed-form or analytical solution may exist under certain restrictive conditions, typically involving the number of bidders and the nature of their private value distributions.

1.  **Specific Value Distributions:** Analytical solutions have been derived for auctions with a small number of players and specific, well-defined probability distributions for bidders' private values. One notable example is the case of a two-player auction where bidder valuations are drawn from **uniform distributions** (http://www.ma.huji.ac.il/~zamir/documents/Uniform_fulltext.pdf). The solution in this specific scenario is complex and tailored to the properties of the uniform distribution.

2.  **Assumed Functional Forms:** In the absence of a general solution method, one approach is to postulate a specific functional form for the equilibrium bidding strategies and then solve for the parameters that satisfy the system of differential equations. This is a "guess and verify" method. For instance, one could assume linear bidding strategies (`g1(b)=αb+β` and `g2(b)=γb+δ`) and attempt to find coefficients that work for a particular setup (https://economics.stackexchange.com/questions/6808/system-of-differential-equations-asymmetric-first-price-auction). This approach yields a solution for a specific case, not a general one.

### Why a General Solution is Elusive:

The system of differential equations in an asymmetric auction is complex because each bidder's strategy depends on the distribution of every other bidder's valuation. As the number of bidders increases or the value distributions become more complex (i.e., not uniform or otherwise well-behaved), the system of equations becomes intractable to solve analytically.

In summary, while specific analytical solutions can be found for highly constrained versions of the asymmetric first-price auction (most commonly, two-player models with uniform distributions), a universal, closed-form solution that applies to any number of bidders with arbitrary value distributions does not appear to exist. Researchers often resort to numerical methods or deriving solutions for very specific, simplified cases.

## What are some specific, well-known examples of asymmetric first-price auctions where the equilibrium bidding functions have been explicitly solved? (e.g., uniform distributions with different upper bounds, exponential distributions with different parameters).



 
 ### Detail the solution and derivation of the equilibrium bidding functions in asymmetric first-price auctions where bidders' valuations are drawn from uniform distributions with different upper bounds (e.g., U[0, b1] and U[0, b2]).

### The Equilibrium Bidding Functions in Asymmetric First-Price Auctions with Uniform Distributions

In a first-price sealed-bid auction, the winner is the participant with the highest bid, and they pay the amount they bid. In an *asymmetric* auction, bidders' private valuations for the item are drawn from different probability distributions. This analysis details the solution and derivation of the equilibrium bidding strategies for two bidders whose valuations are drawn from uniform distributions with different upper bounds.

**1. Model Setup**

Let there be two bidders, Bidder 1 (the "strong" bidder) and Bidder 2 (the "weak" bidder). Their private valuations, `v1` and `v2`, are independently drawn from uniform distributions:
*   Bidder 1's valuation: `v1 ~ U[0, b1]`
*   Bidder 2's valuation: `v2 ~ U[0, b2]`

Without loss of generality, we assume `b1 > b2`, which makes Bidder 1 the strong bidder, as their valuation distribution first-order stochastically dominates that of Bidder 2. The goal is to find the Bayes-Nash equilibrium, which consists of a pair of bidding functions, `β1(v1)` and `β2(v2)`, where each bidder's strategy maximizes their expected payoff given the other's strategy.

**2. Derivation of the Core Differential Equations**

The equilibrium bidding functions can be derived by setting up the expected utility maximization problem for each bidder. It is often easier to work with the inverse bid functions, `ψ1(b)` and `ψ2(b)`, which specify the valuation `v` that corresponds to a given bid `b`.

*   **Bidder 1's Utility:** Bidder 1, with valuation `v1`, chooses a bid `b` to maximize their expected utility `E[U1]`:
    `E[U1(b, v1)] = (v1 - b) * P(β2(v2) < b)`

    The probability that Bidder 2 bids less than `b` is the probability that `v2` is less than the valuation corresponding to bid `b`, which is `ψ2(b)`. Since `v2` is from `U[0, b2]`, this probability is `ψ2(b) / b2`.
    `E[U1(b, v1)] = (v1 - b) * (ψ2(b) / b2)`

*   **Bidder 2's Utility:** Similarly, Bidder 2's expected utility is:
    `E[U2(b, v2)] = (v2 - b) * P(β1(v1) < b) = (v2 - b) * (ψ1(b) / b1)`

To find the optimal bid, we take the first-order condition (the derivative with respect to `b`) and set it to zero.

*   **For Bidder 1:** The derivative of `E[U1]` with respect to `b` is:
    `d/db [ (v1 - b) * (ψ2(b) / b2) ] = (-1/b2) * ψ2(b) + (1/b2) * (v1 - b) * ψ2'(b) = 0`
    In equilibrium, a bidder with valuation `v1` makes the bid `b = β1(v1)`, which implies `v1 = ψ1(b)`. Substituting this into the equation gives the first differential equation:
    **(1) (ψ1(b) - b) * ψ2'(b) = ψ2(b)**

*   **For Bidder 2:** A parallel derivation yields the second differential equation:
    **(2) (ψ2(b) - b) * ψ1'(b) = ψ1(b)**

This system of two coupled ordinary differential equations, along with a set of boundary conditions, defines the equilibrium bidding behavior.

**3. Characterization of the Equilibrium Solution**

The solution to this system is not a simple linear function (unlike the symmetric case where `b1 = b2`). The asymmetry introduces significant complexity, leading to non-linear bidding strategies. The equilibrium has the following structural properties:

*   **Bidding Range:** There is a maximum relevant bid, `b_max`, that will be made in the auction. The weak bidder (`β2`) makes bids over the interval `[0, b_max]`.
*   **Weak Bidder's Strategy:** The weak bidder's strategy `β2(v2)` is a strictly increasing function over their entire valuation range `[0, b2]`. Their maximum possible bid is `b_max = β2(b2)`.
*   **Strong Bidder's Strategy:** The strong bidder's strategy is a two-part function defined by a cutoff valuation `z`, where `b2 < z ≤ b1`.
    *   For valuations `v1` in `[0, z]`, the strong bidder competes directly with the weak bidder, and their bid function `β1(v1)` is strictly increasing.
    *   For valuations `v1` in `(z, b1]`, the strong bidder knows their valuation is higher than the weak bidder's maximum possible valuation (`b2`). They are guaranteed to win and only need to bid enough to beat the weak bidder's highest possible bid. Therefore, for any `v1 > z`, the strong bidder places the same maximum bid: `β1(v1) = b_max`.

The boundary conditions needed to solve the system of differential equations are therefore:
*   `ψ1(0) = 0` and `ψ2(0) = 0` (A bidder with zero valuation bids zero).
*   `ψ2(b_max) = b2` (The weak bidder's maximum valuation corresponds to the maximum bid).
*   `ψ1(b_max) = z` (The strong bidder's cutoff valuation also corresponds to the maximum bid).

**4. The Solution and Its Implications**

While the closed-form solution for the non-linear functions `β1(v)` and `β2(v)` is mathematically complex, we can analyze the bidding behavior for low valuations by finding a linear approximation near `b=0`. This provides key insights into the strategic behavior.

For very low valuations, the bidding functions can be approximated as:
*   **Strong Bidder (1):** `β1(v1) ≈ (1/2) * v1`
*   **Weak Bidder (2):** `β2(v2) ≈ ( v2 / (1 + b2/b1) )`

**Example:** If `b1 = 2` and `b2 = 1`, the strong bidder's strategy is `β1(v1) ≈ 0.5 * v1`, while the weak bidder's strategy is `β2(v2) ≈ v2 / (1 + 1/2) ≈ 0.67 * v2`.

This reveals a crucial strategic insight: **for low valuations, the weak bidder bids a larger fraction of their value than the strong bidder.** The weak bidder bids more aggressively to increase their lower chance of winning. Conversely, the strong bidder "shades" their bid more (bids a smaller fraction of their value) because their higher probability of having a top valuation allows them to be more conservative and still expect to win. For high valuations (`v1 > z`), the strong bidder leverages their advantage by capping their bid at `b_max`, maximizing their profit in cases where they are certain to have the highest value. This complex, non-linear behavior is a direct result of the asymmetry in the bidders' valuation distributions. As one study notes, perturbation analysis can be used to obtain explicit approximations of these equilibrium bids (ResearchGate)

 
 ### Describe the equilibrium bidding strategies for asymmetric first-price auctions when bidders' private values are drawn from exponential distributions with different parameters (e.g., different mean values).

### Equilibrium Bidding Strategies in Asymmetric First-Price Auctions with Exponential Value Distributions

In a first-price, sealed-bid auction, the equilibrium bidding strategy for a bidder is a function that maps their private value for an item to an optimal bid, assuming all other bidders are also bidding optimally. When bidders are asymmetric—meaning their private values are drawn from different probability distributions—the analysis becomes significantly more complex than in the symmetric case. For the specific scenario where two bidders' values are drawn from exponential distributions with different parameters, a closed-form, analytical solution for the bidding strategies is not generally available. Instead, the equilibrium is characterized as the solution to a system of coupled ordinary differential equations that must typically be solved numerically.

#### 1. The Theoretical Framework

Let's consider a first-price auction with two bidders, Bidder 1 and Bidder 2.
*   **Bidder 1's** private value, `v₁`, is drawn from an exponential distribution with parameter `λ₁`. The cumulative distribution function (CDF) is `F₁(v) = 1 - e^(-λ₁v)`, and the probability density function (PDF) is `f₁(v) = λ₁e^(-λ₁v)`. The mean valuation for Bidder 1 is `1/λ₁`.
*   **Bidder 2's** private value, `v₂`, is drawn from an exponential distribution with parameter `λ₂`, with CDF `F₂(v) = 1 - e^(-λ₂v)` and PDF `f₂(v) = λ₂e^(-λ₂v)`. The mean valuation for Bidder 2 is `1/λ₂`.

The asymmetry arises from `λ₁ ≠ λ₂`. The bidder with the lower `λ` parameter has a higher mean valuation and is considered the "strong" bidder, while the bidder with the higher `λ` is the "weak" bidder.

Each bidder `i` seeks to choose a bid `b` to maximize their expected profit, which is the value of winning (`vᵢ - b`) multiplied by the probability of winning. For Bidder 1, the optimization problem is:

`max_b (v₁ - b) * Prob(Bidder 2 bids less than b)`

Let `β₁(v)` and `β₂(v)` be the equilibrium bid functions for Bidder 1 and 2, respectively. These functions are strictly increasing. We can define their inverses as `y₁(b) = β₁⁻¹(b)` and `y₂(b) = β₂⁻¹(b)`, which represent the value a bidder must have to place a bid of `b`.

The optimization problems for the two bidders lead to a pair of first-order conditions. These conditions can be expressed as a system of coupled differential equations for the inverse bid functions [ (users.ssc.wisc.edu)](https://users.ssc.wisc.edu/~dquint/econ805%202007/econ%20805%20lecture%209.pdf). The general form of this system is:

1.  `y₁'(b) = (y₁(b) - b) * [f₂(y₂(b)) / F₂(y₂(b))]`
2.  `y₂'(b) = (y₂(b) - b) * [f₁(y₁(b)) / F₁(y₁(b))]`

#### 2. Application to Exponential Distributions

For the exponential distribution, the term `f(y) / F(y)` (known as the hazard rate) is `λe^(-λy) / (1 - e^(-λy))`. Substituting this into the system gives:

1.  `y₁'(b) = (y₁(b) - b) * [λ₂e^(-λ₂y₂(b)) / (1 - e^(-λ₂y₂(b)))]`
2.  `y₂'(b) = (y₂(b) - b) * [λ₁e^(-λ₁y₁(b)) / (1 - e^(-λ₁y₁(b)))]`

This system is subject to the boundary condition that a bidder with a value of zero will bid zero: `β₁(0) = β₂(0) = 0`, which implies `y₁(0) = y₂(0) = 0`.

#### 3. Characteristics of the Equilibrium and Lack of a Closed-Form Solution

This system of differential equations does not have a known general closed-form (analytical) solution. As a result, the equilibrium bid functions must be computed using numerical methods [ (capcp.la.psu.edu)](https://capcp.la.psu.edu/wp-content/uploads/sites/11/2020/07/NumericalSolutions.pdf).

Despite the absence of a simple formula, the properties of the equilibrium strategies can be described:

*   **Existence and Uniqueness**: An equilibrium in strictly increasing bid functions is known to exist and is unique.
*   **Bidding Behavior**: The strong bidder (with the higher average valuation) will generally bid more aggressively than in a symmetric auction. Conversely, the weak bidder may bid more or less aggressively depending on the specific parameters, as they must balance the lower chance of winning with the potential payoff.
*   **No Simple "Shading" Rule**: Unlike in symmetric auctions where a bidder with value `v` bids the expected value of the second-highest valuation given their own is the highest, no such simple rule applies here. The optimal bid for one player is intricately linked to the entire value distribution of the other player.

In summary, the equilibrium bidding strategies in a first-price auction with asymmetrically distributed exponential values are defined by the solution to a system of coupled differential equations. Because this system lacks an analytical solution, the strategies are characterized by their qualitative properties and must be determined through numerical computation.

 
 ### Identify and summarize other well-known, specific examples of asymmetric first-price auctions with explicitly solved equilibrium bidding functions, beyond the standard uniform and exponential distribution cases. This could include cases with power-law distributions or other named distributions.

Based on the provided web search results, it is not possible to identify well-known, specific examples of asymmetric first-price auctions with explicitly solved equilibrium bidding functions for distributions other than the uniform case. The search results focus predominantly on the analysis of asymmetric auctions where bidders' valuations are drawn from uniform distributions.

Here is a summary of the findings from the provided text:

*   **Uniform Distributions:** The documents extensively discuss asymmetric first-price auctions with uniform distributions. One source provides a complex analytical solution for a bidder's inverse bid function in a specific uniform distribution scenario [v1, v1] = [0, 1], [v2, v2] = [β, β + ε] (http://www.ma.huji.ac.il/~zamir/documents/Uniform_fulltext.pdf). Another paper proposes a necessary and sufficient condition for the existence of a linear bid equilibrium when two bidders' values are drawn from asymmetric uniform distributions (https://www.researchgate.net/publication/228319685_Linear_Bid_in_Asymmetric_First-Price_Auctions).

*   **General Approximations:** One result mentions the use of "perturbation analysis to obtain explicit approximations of the equilibrium bids in asymmetric first-price auctions with n bidders, in which bidders' valuations are independently drawn from different distribution functions" (https://www.researchgate.net/publication/228319685_Linear_Bid_in_Asymmetric_First-Price_Auctions). However, it does not provide the explicit solutions or name the specific distributions (like power-law, etc.) for which these approximations are derived.

The provided search results do not contain any mention of explicitly solved equilibrium bidding functions for asymmetric first-price auctions involving power-law distributions or any other specifically named distributions beyond the uniform case. Therefore, the information required to answer the sub-topic question is not present in the supplied context.

 
 ### In cases where an analytical or closed-form solution cannot be found, what are the primary numerical and computational methods used to approximate the equilibrium bidding strategies for asymmetric first-price auctions?

When an analytical or closed-form solution is not feasible for asymmetric first-price auctions, researchers and practitioners turn to numerical and computational methods to approximate the equilibrium bidding strategies. Based on the provided research, the primary methods involve robust, automated algorithms that compute the inverse equilibrium bid functions.

Key methods and approaches include:

*   **Numerical Approximation Algorithms:** The core of solving these complex auctions involves numerical approximation. These methods are extensions of earlier work, such as the methods developed by Marshall, Meurer, Richard, and Stromquist (1994) and Gayle (2004). These algorithms are designed to numerically solve first-price auction problems where bidders draw independent private values from different (heterogeneous) distributions (ResearchGate).

*   **Iterative Processes:** To find specific components of the equilibrium, such as the probabilities of entry for each bidder, iterative methods are employed alongside the main numerical approximation for the bids themselves (ResearchGate).

*   **Automated Algorithms for Inverse Bid Functions:** Researchers have developed powerful and fully automated algorithms specifically to compute the inverse equilibrium bid functions for these asymmetric auctions. These tools can handle auctions with bidders who draw Independent and Private Values from various distributions and can also account for factors like a reserve price (capcp.la.psu.edu).

In essence, the field relies on sophisticated computer programs that implement numerical approximation and iteration to find the equilibrium bids in scenarios too complex for traditional analytical solutions (ResearchGate; capcp.la.psu.edu).


## Citations
- https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/777164baec3d203bc6da462488d371e0_MIT14_12F12_chapter15.pdf
- https://math.stackexchange.com/questions/3423092/first-price-auction-symmetric-equilibrium-derivation
- https://cs.brown.edu/courses/cs1951k/lectures/2020/first_price_auctions.pdf
- https://homepages.math.uic.edu/~marker/stat473-F14/auctions.pdf
- https://www.chegg.com/homework-help/questions-and-answers/consider-independent-private-value-first-price-sealed-bid-auction-two-bidders-bidder-s-pri-q111433477
- https://math.arizona.edu/~rbt/auctions.PDF
- https://eml.berkeley.edu/~mcfadden/eC103_f03/auctionlect.pdf
- https://www.chegg.com/homework-help/questions-and-answers/asymmetric-bidders-consider-two-bidders-b-independent-private-values-suppose-bidders-asymm-q103391564
- https://economics.stackexchange.com/questions/6808/system-of-differential-equations-asymmetric-first-price-auction
- https://www.chegg.com/homework-help/questions-and-answers/1-consider-independent-private-value-first-price-sealed-bid-auction-two-bidders-bidder-s-p-q72172702
- http://www.ma.huji.ac.il/~zamir/documents/Uniform_fulltext.pdf
- https://www.asc.ohio-state.edu/ye.45/Econ816/Hafalir-Vijay.pdf
- https://capcp.la.psu.edu/wp-content/uploads/sites/11/numericalanalysis.pdf
- https://www.researchgate.net/publication/228319685_Linear_Bid_in_Asymmetric_First-Price_Auctions
- https://bugarinmauricio.com/wp-content/uploads/2017/06/krishna-auction-theory-caps1a4.pdf
- https://economics.stackexchange.com/questions/14412/derivation-of-equilibrium-strategy-in-1st-price-auction
- http://www.diva-portal.org/smash/get/diva2:1245417/FULLTEXT01.pdf
- https://capcp.la.psu.edu/wp-content/uploads/sites/11/2020/07/NumericalSolutions.pdf
- https://bpb-us-w2.wpmucdn.com/sites.wustl.edu/dist/3/2139/files/2019/09/basic-auctions.pdf
- https://math.stackexchange.com/questions/1385728/system-of-differential-equations-asymmetric-first-price-auction
- https://economics.uwo.ca/people/zheng_docs/1stpriceasym.pdf
- https://www.uoguelph.ca/economics/repec/workingpapers/2015/2015-02.pdf
- https://users.ssc.wisc.edu/~dquint/econ805%202007/econ%20805%20lecture%209.pdf
- https://hanzhezhang.github.io/teaching/Chicago_ECON207/207sol_auction.pdf
- https://cs.brown.edu/courses/csci1440/lectures/2025/first_price_auctions.pdf
- https://www.researchgate.net/publication/228592056_Numerical_Analysis_of_Asymmetric_First_Price_Auctions_with_Reserve_Prices
- https://www.researchgate.net/publication/226095950_Numerical_Solutions_of_Asymmetric_First-Price_Independent_Private_Values_Auctions
- https://www.cirje.e.u-tokyo.ac.jp/research/workshops/micro/documents/March20.pdf
- http://comlabgames.com/47-901/lectures/1%20Sealed%20Bid%20Auctions.pdf
- https://www.econgraphs.org/explanations/game/auctions/first_price_auctions
- https://scholar.harvard.edu/files/maskin/files/asymmetric_auctions.pdf
