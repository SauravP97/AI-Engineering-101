# Deep Research Report

## Table of Contents 
- Explain the methodology of the Mean-Variance model, detailing its approach to risk measurement using variance, return prediction based on historical means, and the process of constructing the efficient frontier for optimal asset allocation.
- Investigate the core theoretical assumptions underpinning the Mean-Variance model. This includes the assumption of normal distribution for asset returns, the concept of rational investors (risk aversion), and the conditions of frictionless markets (e.g., no transaction costs or taxes).
- "Detail the Black-Litterman model's theoretical foundation as an evolution of Mean-Variance Optimization. Explain its core components, including the derivation of market-implied equilibrium returns and the mathematical framework used to incorporate subjective investor views to produce a new, blended set of expected returns.",
- "Investigate the application of deep learning models (LSTMs, Recurrent Neural Networks) in asset allocation, focusing on how they are used for return prediction and risk measurement. Detail the mechanisms by which these models capture complex, non-linear market dynamics.",
- For the Mean-Variance Optimization model, detail its methodologies for: 1) Risk Measurement (e.g., use of covariance matrix), 2) Return Prediction (e.g., reliance on historical sample means), and 3) the resulting Asset Allocation output (i.e., the efficient frontier). Analyze its theoretical foundations in Modern Portfolio Theory and the practical implications, such as sensitivity to input errors.

## Report 
## In the field of FinTech, machine learning algorithms are now widely applied to asset allocation and investment decisions. Examples include classic models like Mean-Variance and Black-Litterman, as well as emerging deep learning models. While these models have shown certain advantages under different market conditions, each also has its limitations. For instance, the Mean-Variance model assumes asset returns follow a normal distribution, which often doesn't align with actual market conditions. The Black-Litterman model relies on subjective view inputs, introducing a degree of subjectivity. Although deep learning models can handle complex non-linear relationships, they suffer from poor interpretability. So, what are the core differences between these various models in terms of risk measurement, return prediction, and asset allocation? And is it possible to combine their strengths to build a more general-purpose and effective modeling framework?



## Analyze the Mean-Variance model's methodology for risk measurement (e.g., variance), return prediction (e.g., historical means), and asset allocation. Investigate its core assumptions, such as the normal distribution of asset returns, and the practical limitations arising from these assumptions in real-world market conditions.



 
 ### Explain the methodology of the Mean-Variance model, detailing its approach to risk measurement using variance, return prediction based on historical means, and the process of constructing the efficient frontier for optimal asset allocation.

### The Methodology of the Mean-Variance Model

The Mean-Variance model, a cornerstone of Modern Portfolio Theory (MPT), provides a systematic framework for constructing diversified investment portfolios. Its methodology is centered on the trade-off between risk and return, allowing investors to identify an optimal asset allocation that aligns with their personal risk tolerance. The model operates by measuring risk through variance, predicting returns using historical averages, and then plotting a set of optimal portfolios known as the Efficient Frontier.

#### 1. Risk Measurement Using Variance

In mean-variance analysis, "risk" is quantified by the statistical measure of variance (or its square root, the standard deviation). Variance measures the volatility or dispersion of an asset's returns around its average (mean) return. A higher variance indicates that the asset's returns have been more spread out and are thus considered riskier, while a lower variance suggests more stable and predictable returns.

The model extends this concept to an entire portfolio. The risk of a portfolio is not simply the weighted average of the individual asset variances. It also critically depends on the **covariance** between the assets in the portfolio. Covariance measures how two assets move in relation to each other.

*   **Positive Covariance:** The assets' returns tend to move in the same direction.
*   **Negative Covariance:** The assets' returns tend to move in opposite directions.

By combining assets with low or negative covariance, an investor can achieve diversification. This means the overall portfolio variance can be lower than the weighted average of the individual asset variances, as the poor performance of one asset may be offset by the strong performance of another. This is the mathematical principle behind the adage, "Don't put all your eggs in one basket."

#### 2. Return Prediction Based on Historical Means

The "mean" in the Mean-Variance model refers to the expected return. The model typically uses the historical average (mean) return of an asset as the primary predictor for its future expected return. This is a foundational assumption of the model: that past performance is indicative of future results.

The expected return of a portfolio is calculated as the weighted average of the expected returns of the individual assets it contains. For example, for a two-asset portfolio:

*Expected Portfolio Return = (Weight of Asset A * Expected Return of Asset A) + (Weight of Asset B * Expected Return of Asset B)*

While simple to calculate, this reliance on historical data is a significant limitation, as past returns do not guarantee future returns.

#### 3. Constructing the Efficient Frontier

The culmination of the mean-variance methodology is the construction of the Efficient Frontier. This is a graphical representation of all possible portfolios that are optimally diversified. The process is as follows:

1.  **Input Gathering:** Collect the necessary data for a universe of potential investments:
    *   The expected (mean) return for each asset.
    *   The variance (or standard deviation) of each asset.
    *   The covariance between every pair of assets.

2.  **Portfolio Generation:** Using these inputs, an optimization algorithm generates a vast number of possible portfolios by combining the assets in every conceivable weighting.

3.  **Risk-Return Plotting:** Each generated portfolio's total expected return and total variance (risk) are calculated and plotted on a graph, with risk on the x-axis and expected return on the y-axis.

4.  **Identifying the Frontier:** The resulting cloud of points represents all possible portfolios. The **Efficient Frontier** is the upward-curving line that forms the upper-left boundary of this set of points.

Any portfolio that lies on the Efficient Frontier is considered "optimal" because it offers the highest possible expected return for its level of risk. Conversely, for any given expected return, the portfolio on the frontier has the lowest possible risk. Portfolios below the frontier are sub-optimal because another portfolio exists with either a higher return for the same risk or lower risk for the same return.

Finally, an investor chooses a single portfolio from the Efficient Frontier that best matches their individual risk tolerance and return objectives. This framework allows investors to systematically allocate assets to achieve specific risk-return targets and provides a benchmark for evaluating portfolio performance [Source: linkedin.com/pulse/mean-variance-framework-combining-investments-maqsood-acca-socpa-26wqf]. The analysis is a core part of MPT, which assumes that investors, given complete information, will make rational decisions to maximize their return for a chosen level of risk [Source: investopedia.com/terms/m/meanvariance-analysis.asp]. The visual representation of the Efficient Frontier, often shown with and without a risk-free asset, is central to mean-variance optimization [Source: columbia.edu/~mh2078/FoundationsFE/MeanVariance-CAPM.pdf].

 
 ### Investigate the core theoretical assumptions underpinning the Mean-Variance model. This includes the assumption of normal distribution for asset returns, the concept of rational investors (risk aversion), and the conditions of frictionless markets (e.g., no transaction costs or taxes).

The Mean-Variance model, the cornerstone of Modern Portfolio Theory (MPT) developed by Harry Markowitz, is built upon several core theoretical assumptions that allow for the mathematical optimization of portfolios. These assumptions simplify the complexities of financial markets to create a workable framework for asset allocation. The primary assumptions include the normality of asset returns, the rationality and risk aversion of investors, and the existence of frictionless markets.

### 1. Normal Distribution of Asset Returns

A central assumption of the Mean-Variance model is that the returns of assets are normally distributed (or follow a similar elliptical distribution). This statistical assumption is critical because it allows the entire probability distribution of returns for a portfolio to be perfectly described by just two parameters:

*   **Mean (Expected Return):** The average return an investor expects to receive.
*   **Variance (or Standard Deviation):** A measure of the dispersion or volatility of returns, which serves as the model's proxy for risk.

By assuming a normal distribution, the model simplifies the complex reality of market returns, making the mathematics of portfolio optimization tractable. It implies that returns are symmetric around the mean and that extreme, or "tail," events are highly unlikely. However, this is one of the most criticized aspects of the model, as empirical evidence often shows that financial returns are not perfectly normal and can exhibit "fat tails" (kurtosis) and skewness, meaning extreme events occur more frequently than the model predicts.

### 2. Rational and Risk-Averse Investors

The model's framework is grounded in a specific understanding of investor behavior. It assumes that all investors are:

*   **Rational:** They aim to maximize their economic utility. In this context, it means they seek to achieve the highest possible return for their chosen level of risk.
*   **Risk-Averse:** This is a foundational concept within the model. Markowitz and subsequent MPT proponents operate under the assumption that investors are fundamentally risk-averse (https://www.vestinda.com/portfolio/what-is-modern-portfolio-theory-and-its-key-assumptions, https://www1.stjameswinery.com/virtual-library/hSVSiM/271011/Markowitz%20Portfolio%20Theory.pdf). This does not mean they avoid risk entirely, but rather that for a given level of expected return, they will always prefer the portfolio with the least amount of risk. Conversely, if two portfolios have the same level of risk, a rational, risk-averse investor will always choose the one with the higher expected return. This preference for less risk for the same return is what drives the search for an "optimal portfolio" (https://www1.stjameswinery.com/virtual-library/hSVSiM/271011/Markowitz%20Portfolio%20Theory.pdf).

### 3. Frictionless Markets

The Mean-Variance model operates in an idealized economic environment, assuming markets are "frictionless." This means that there are no external factors or costs that hinder the trading process. The key conditions of a frictionless market include:

*   **No Transaction Costs:** The model assumes that investors can buy and sell assets without incurring any brokerage fees, commissions, or other trading costs.
*   **No Taxes:** The analysis of returns is done on a pre-tax basis, meaning the impact of capital gains or dividend taxes on portfolio performance is ignored.
*   **Perfect Liquidity and Divisibility:** Assets are assumed to be perfectly liquid and infinitely divisible, meaning any quantity of an asset, no matter how small, can be bought or sold at the prevailing market price at any time.
*   **No Single Investor Can Influence Prices:** The market is assumed to be composed of many investors, none of whom can individually affect the market price of an asset by their buying or selling activities.

These assumptions of a perfect market simplify the model significantly, but they also represent a departure from real-world conditions where transaction costs, taxes, and market liquidity are important considerations for investors.

## Examine the Black-Litterman model as an evolution of Mean-Variance. Detail its approach to incorporating subjective investor views for return prediction, its risk measurement framework, and the resulting asset allocation process. What are the main challenges and limitations associated with the subjectivity of its inputs?



 
 ### "Detail the Black-Litterman model's theoretical foundation as an evolution of Mean-Variance Optimization. Explain its core components, including the derivation of market-implied equilibrium returns and the mathematical framework used to incorporate subjective investor views to produce a new, blended set of expected returns.",

### The Black-Litterman Model: A Theoretical Evolution of Mean-Variance Optimization

The Black-Litterman model is a sophisticated portfolio allocation tool that serves as a significant evolution of the foundational Mean-Variance Optimization (MVO) framework, also known as Modern Portfolio Theory (MPT) (Investopedia, 2023). Developed by Fischer Black and Robert Litterman in 1990, the model addresses the primary practical challenges of MVO by providing a more stable and intuitive method for generating expected returns, which are a critical input for the optimization process (Investopedia, 2023).

The core innovation of the Black-Litterman model is its ability to blend the collective wisdom of the market (equilibrium returns) with an individual investor's unique insights or views to produce a refined set of expected returns for asset allocation (Investglass, 2023).

---

### Core Components and Mathematical Framework

The Black-Litterman process can be broken down into three main components: deriving market-implied returns, specifying investor views, and combining these two elements to create a new, blended set of expected returns.

#### 1. Market-Implied Equilibrium Returns: The Starting Point

A major drawback of traditional MVO is its high sensitivity to the expected returns input. Small changes in these estimates can lead to large, often unintuitive, swings in asset weights, frequently resulting in concentrated, undiversified portfolios. The Black-Litterman model overcomes this by forgoing the need for the user to input a complete set of expected returns. Instead, it starts with a neutral, market-implied set of returns.

This is achieved through a process of **reverse optimization** (FE Training, n.d.). The model assumes that the existing global market portfolio, with its observed market capitalization weights, is itself an optimal portfolio. Working backward from this assumption, it calculates the set of expected returns that would make the current market weights optimal.

The derivation for the **Implied Equilibrium Return Vector (Π)** is as follows:

**Π = λΣw_mkt**

Where:
*   **Π (Pi)**: The vector of implied excess equilibrium returns for each asset. This represents the market's consensus view.
*   **λ (Lambda)**: A scalar representing the market's coefficient of risk aversion. It is calculated as the market's expected return divided by its variance.
*   **Σ (Sigma)**: The covariance matrix of excess returns for the assets.
*   **w_mkt**: The vector of market capitalization weights for the assets (Duke University, n.d.).

This equilibrium return vector serves as a neutral and stable starting point, reflecting the market's collective forecast before any individual views are considered.

#### 2. Incorporating Subjective Investor Views

The model then provides a structured framework for an investor to express their specific, subjective forecasts about the market (FE Training, n.d.). These views do not need to be comprehensive; an investor can express opinions on the absolute or relative performance of just a few assets.

These views are mathematically formulated using the following components:

*   **P (View Matrix)**: A matrix that identifies the assets involved in each view. For example, if an investor believes Asset A will outperform Asset B by 2%, the row in the P matrix corresponding to this view would have a `+1` in the column for Asset A and a `-1` in the column for Asset B.
*   **Q (Expected Returns Vector for Views)**: A vector containing the specific return forecast for each view. In the example above, the corresponding entry in the Q vector would be `0.02`.
*   **Ω (Omega)**: A diagonal covariance matrix representing the uncertainty or confidence level of each view. The diagonal elements are the variances of the error terms for each view. A smaller variance signifies higher confidence in a particular view, giving it more weight in the final calculation (Duke University, n.d.).

#### 3. The New, Blended Set of Expected Returns

The final and most crucial step is the mathematical combination of the neutral market equilibrium returns (the prior) with the investor's subjective views (the likelihood). The model uses a Bayesian framework to produce a new, blended vector of expected returns (the posterior) that is a weighted average of the two inputs.

The formula for the **New Combined Return Distribution (E[R])** is:

**E[R] = [ (τΣ)⁻¹ + P'Ω⁻¹P ]⁻¹ [ (τΣ)⁻¹Π + P'Ω⁻¹Q ]**

Where:
*   **E[R]**: The new, blended vector of expected returns.
*   **τ (Tau)**: A scalar representing the uncertainty in the prior equilibrium returns.
*   **Σ**: The covariance matrix of asset returns.
*   **P**: The view matrix.
*   **Ω**: The uncertainty matrix for the views.
*   **Π**: The implied equilibrium return vector.
*   **Q**: The vector of returns associated with the views (Duke University, n.d.; CIMS NYU, 2021).

The intuition behind this formula is that the final expected return for each asset is a weighted average of the market-implied return and the investor's view. The weight assigned to the investor's view is directly proportional to the confidence expressed in that view (i.e., inversely proportional to the variance in Ω). If an investor expresses a view with high confidence, the resulting expected return will be tilted more strongly towards their view and away from the market equilibrium (FE Training, n.d.).

This new vector of expected returns, E[R], can then be used as the input for a standard mean-variance optimizer to calculate the final, optimized portfolio weights. This process results in portfolios that are better diversified, more stable, and more intuitive, as they reflect both the wisdom of the market and the specific insights of the investor.

### References
*   CIMS NYU. (2021). The Black-Litterman Model. Retrieved from https://cims.nyu.edu/~ritter/kolm2021black.pdf
*   Duke University. (n.d.). A STEP-BY-STEP GUIDE TO THE BLACK-LITTERMAN MODEL. Retrieved from https://people.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf
*   FE Training. (n.d.). Black-Litterman Model. Retrieved from https://www.fe.training/free-resources/portfolio-management/black-litterman-model/
*   Investglass. (2023). Mastering Portfolio Optimization with the Black-Litterman Model. Retrieved from https://www.investglass.com/mastering-portfolio-optimization-with-the-black-litterman-model/
*   Investopedia. (2023). Black-Litterman Model: Definition, Example, Pros & Cons. Retrieved from https://www.investopedia.com/terms/b/black-litterman_model.asp

## Investigate the application of deep learning models (e.g., LSTMs, Recurrent Neural Networks) in asset allocation. Focus on how they handle return prediction and risk measurement by capturing complex, non-linear market dynamics. Critically evaluate the major limitation of these models: their 'black-box' nature and lack of interpretability.



 
 ### "Investigate the application of deep learning models (LSTMs, Recurrent Neural Networks) in asset allocation, focusing on how they are used for return prediction and risk measurement. Detail the mechanisms by which these models capture complex, non-linear market dynamics.",

### Deep Learning Models in Asset Allocation: Predicting Returns and Measuring Risk

Recurrent Neural Networks (RNNs) and their advanced variant, Long Short-Term Memory (LSTM) models, are increasingly being applied to the complex task of asset allocation. Their primary advantage lies in their ability to analyze time-series data, making them particularly well-suited for financial markets, which are characterized by inherent volatility and non-linearity (https://arxiv.org/html/2505.05325v1). These deep learning models are primarily used for two key functions within asset allocation: return prediction and risk measurement.

#### Return Prediction

The core of any asset allocation strategy is forecasting the expected return of various assets. LSTMs are used to model and predict future stock market behavior and obtain expected returns (https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2151159).

An LSTM model is trained on historical sequences of market data, which can include price, trading volume, and even sentiment analysis data. By processing this information sequentially, the model learns the underlying patterns and temporal dependencies that may indicate future price movements. This process allows the model to forecast future prices, which in turn can be used to calculate expected returns for different assets. These return forecasts then inform the asset allocation decision, guiding how capital should be distributed across various investment options.

#### Risk Measurement and Management

Beyond predicting returns, understanding and managing risk is critical. Deep learning models are also applied to predict market volatility, a common proxy for investment risk (https://www.mdpi.com/2673-9909/5/3/76). By forecasting periods of high or low volatility, investors can adjust their portfolios accordingly. For instance, an RNN-based model can be used to predict short-term market fluctuations, which can inform dynamic hedging strategies to mitigate potential losses (https://www.researchgate.net/publication/387526399_Dynamic_Asset_Allocation_Using_Recurrent_Neural_Networks_RNNs). This forward-looking measure of risk is a significant advantage over traditional risk models that often rely on historical volatility alone.

#### Mechanisms for Capturing Market Dynamics

The effectiveness of RNNs and LSTMs in financial markets stems from their unique architecture, which is designed to capture complex, non-linear dynamics and long-range patterns in sequential data (https://arxiv.org/html/2505.05325v1).

1.  **Sequential Memory:** Unlike traditional neural networks that treat each data point independently, RNNs possess a form of memory. The output from one step is fed back as an input to the next step, allowing the network to retain information about past events. This is crucial in financial markets where past events, such as a previous day's closing price or a central bank announcement, can influence future prices.

2.  **Capturing Non-Linearity:** Financial markets are not linear; the relationship between inputs (e.g., economic indicators) and outputs (e.g., stock prices) is complex and constantly changing. The activation functions within the neural network layers allow RNNs and LSTMs to model these intricate, non-linear relationships effectively (https://pmc.ncbi.nlm.nih.gov/articles/PMC12334627/).

3.  **Long-Term Dependencies (LSTM):** A limitation of simple RNNs is their difficulty in retaining information over very long sequences (the "vanishing gradient problem"). LSTMs were specifically designed to overcome this. They incorporate a "cell state" and a series of "gates" (input, forget, and output gates). These gates regulate the flow of information, allowing the model to learn which data from the past is important to remember and which can be forgotten. This mechanism is what enables LSTMs to capture the long-range patterns and temporal dependencies that are characteristic of stock market data (https://arxiv.org/html/2505.05325v1). This adaptability is key to adjusting to dynamic market conditions and making more accurate predictions for both returns and risk (https://pmc.ncbi.nlm.nih.gov/articles/PMC12334627/).

## Conduct a comparative analysis of the Mean-Variance, Black-Litterman, and deep learning models. Create a structured comparison across three specific dimensions: 1) Risk Measurement techniques, 2) Return Prediction methodologies, and 3) the final Asset Allocation output. Highlight the core differences in their theoretical foundations and practical implications for each dimension.



 
 ### For the Mean-Variance Optimization model, detail its methodologies for: 1) Risk Measurement (e.g., use of covariance matrix), 2) Return Prediction (e.g., reliance on historical sample means), and 3) the resulting Asset Allocation output (i.e., the efficient frontier). Analyze its theoretical foundations in Modern Portfolio Theory and the practical implications, such as sensitivity to input errors.

### Mean-Variance Optimization: Methodologies, Foundations, and Practical Implications

The Mean-Variance Optimization (MVO) model, introduced by Harry Markowitz in 1952, remains a foundational element of Modern Portfolio Theory (MPT) for efficient asset allocation. It provides a mathematical framework for constructing portfolios by balancing expected returns against risk.

#### **1. Methodologies of the MVO Model**

The MVO model's methodology is centered on three key components: risk measurement, return prediction, and the resulting asset allocation.

**a) Risk Measurement: The Covariance Matrix**
The model quantifies portfolio risk as the variance (or standard deviation) of the portfolio's returns. This is not simply a weighted average of individual asset variances. Instead, it critically relies on the **covariance matrix**. This matrix is essential as it measures not only the individual volatility of each asset but also how each asset's returns move in relation to every other asset in the portfolio. The only data required to conduct this analysis are the mean returns and the covariance matrix. This comprehensive approach allows the model to capture the benefits of diversification; by combining assets that are not perfectly positively correlated, a portfolio's overall variance can be lower than the sum of its parts.

**b) Return Prediction: Historical Sample Means**
For predicting future returns, the MVO model traditionally relies on the **historical sample means** of asset returns. This approach uses past average returns as the primary estimate for future expected returns. While straightforward to calculate, this methodology is a significant point of criticism. It assumes that past performance is indicative of future results, an assumption that may not hold true, especially in changing market conditions.

**c) Asset Allocation Output: The Efficient Frontier**
The output of the Mean-Variance Optimization is the **efficient frontier**. This is a curve on a risk-return graph representing the set of optimal portfolios. Each point on the frontier corresponds to a portfolio that offers the highest possible expected return for a given level of risk (measured by variance). Conversely, for any given level of expected return, the portfolio on the efficient frontier has the lowest possible risk. An investor's personal risk tolerance determines which point on the frontier is most suitable for them. When a risk-free asset is introduced, the efficient frontier becomes a straight line known as the Capital Allocation Line (CAL), which is tangent to the curved efficient frontier of risky assets. This tangency point represents the optimal portfolio of risky assets for all investors, according to the model.

#### **2. Theoretical Foundations and Practical Implications**

**a) Foundations in Modern Portfolio Theory (MPT)**
MVO is the operational model that brings the concepts of MPT to life. MPT's central tenet is that investors are risk-averse and that diversification can reduce portfolio risk without sacrificing returns. MVO formalizes this by using the covariance matrix to mathematically construct diversified portfolios that optimize this risk-return trade-off, resulting in the efficient frontier. The model is also a precursor to the Capital Asset Pricing Model (CAPM), which builds upon MVO's framework by assuming that if every investor is a mean-variance optimizer, they will all hold a combination of the risk-free asset and the same "tangency portfolio" of risky assets.

**b) Practical Implications and Sensitivity to Input Errors**
Despite its theoretical elegance, MVO has significant practical limitations, most notably its high sensitivity to input errors. This issue is sometimes referred to as the "optimization enigma". The model's outputs (the asset weights) are extremely sensitive to small changes or estimation errors in the inputs—the expected returns (sample means) and the covariance matrix. Since these inputs are merely estimates of future conditions based on historical data, any inaccuracies can lead to dramatically different and often unstable or unintuitive portfolio allocations. This "estimation error" challenges the real-world applicability of the model, as portfolios constructed using MVO can perform poorly out-of-sample if the historical inputs do not accurately reflect future market behavior. Consequently, the traditional estimation of the covariance matrix is a subject of ongoing research and challenge within portfolio allocation.

## Explore the potential for creating a hybrid modeling framework that combines the strengths of classic and deep learning models. Research existing or proposed methodologies that integrate, for example, the theoretical rigor of Black-Litterman with the predictive power of deep learning models to improve robustness, interpretability, and overall effectiveness in asset allocation.




## Citations
- https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2151159
- https://en.wikipedia.org/wiki/Modern_portfolio_theory
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12334627/
- https://www.fe.training/free-resources/portfolio-management/black-litterman-model/
- https://www1.stjameswinery.com/virtual-library/hSVSiM/271011/Markowitz%20Portfolio%20Theory.pdf
- https://www.princeton.edu/~markus/teaching/Fin501/07Lecture.pdf
- https://www.diva-portal.org/smash/get/diva2:1060405/FULLTEXT01.pdf
- https://www.mdpi.com/2673-9909/5/3/76
- https://arxiv.org/html/2505.05325v1
- https://people.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf
- https://www.reddit.com/r/HomeworkHelp/comments/1jvvyer/university_finance_portfolio_theory_help_with/
- https://www.columbia.edu/~mh2078/FoundationsFE/MeanVariance-CAPM.pdf
- https://cims.nyu.edu/~ritter/kolm2021black.pdf
- https://www.vestinda.com/portfolio/what-is-modern-portfolio-theory-and-its-key-assumptions
- https://www.researchgate.net/publication/387526399_Dynamic_Asset_Allocation_Using_Recurrent_Neural_Networks_RNNs
- https://www.linkedin.com/pulse/mean-variance-framework-combining-investments-maqsood-acca-socpa-26wqf
- https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios
- https://www.youtube.com/watch?v=zVVphMSZKBM
- https://www.investopedia.com/terms/m/meanvariance-analysis.asp
- https://www.investopedia.com/terms/b/black-litterman_model.asp
- https://www.investglass.com/mastering-portfolio-optimization-with-the-black-litterman-model/
