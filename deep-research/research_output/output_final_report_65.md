# Deep Research Report

## Table of Contents 
- Observability Analysis of Phenotypic Traits: Using the state-space models, conduct a formal observability analysis to determine which phenotypic traits are theoretically measurable from the defined sensor outputs. This involves assessing the observability matrix and identifying which states can be uniquely determined from the output data.
- "Investigate and design an optimal estimation system (e.g., Kalman filter, particle filter) for real-time tracking of phenotypic traits during 3D reconstruction. This includes defining the state-space model for trait dynamics, the measurement model linking states to sensor data, and comparing filter performance for this specific application.",
- Detail the integration of the proposed estimation system and control strategy into a cohesive, closed-loop system. Explain how the real-time state estimates from the filter will be used as feedback for the controller, and how the controller's actions will, in turn, provide better data to improve the accuracy of the estimation, creating a synergistic feedback loop."
- In-depth comparison of the proposed control theory-based approach with machine learning methods (specifically CNNs and GANs). This analysis should evaluate performance metrics, computational complexity, data requirements, and applicability to the problem domain.

## Report 
## As an agricultural engineering researcher focusing on 3D reconstruction and phenotypic analysis of crop grains, please develop a design report utilizing modern control theory, alongside other relevant theoretical methods and models, for the tasks of modeling, analysis, and design pertinent to my research area.



## State-of-the-Art Review and Problem Formulation: Conduct a comprehensive review of current 3D reconstruction and phenotypic analysis techniques for crop grains, identifying their limitations. Formulate the research problem within the framework of modern control theory, defining the system, inputs, outputs, and performance objectives for the design report.



## System Modeling and Identification: Develop dynamic and state-space models representing the geometric and textural properties of crop grains for 3D reconstruction. Investigate system identification techniques to estimate model parameters from imaging sensor data, forming the basis for control-theoretic analysis.



## Control-Theoretic Analysis of Phenotypic Traits: Apply modern control theory concepts, such as observability and controllability, to analyze the developed grain models. The goal is to determine which phenotypic traits are theoretically measurable and how accurately they can be estimated from the available sensor information.



 
 ### Observability Analysis of Phenotypic Traits: Using the state-space models, conduct a formal observability analysis to determine which phenotypic traits are theoretically measurable from the defined sensor outputs. This involves assessing the observability matrix and identifying which states can be uniquely determined from the output data.

### Observability Analysis of Phenotypic Traits Using State-Space Models

**1. Introduction to State-Space Models for Phenotypic Analysis**

In the context of modern plant phenotyping, state-space models provide a powerful mathematical framework for describing the dynamics of plant growth and development over time. These models represent the internal state of a system (the plant) through a set of variables, known as state variables, and relate them to external inputs and measurable outputs.

A standard linear time-invariant state-space model is defined by two equations:
1.  **State Equation:** `ẋ = Ax + Bu`
2.  **Output Equation:** `y = Cx + Du`

Where:
*   `x` is the **state vector**, representing the set of key phenotypic traits of interest. These are the internal, often unmeasurable, characteristics of the plant we want to understand. Examples include biomass, nitrogen content, water potential, or photosynthetic efficiency.
*   `u` is the **input vector**, representing external factors that influence the plant's growth, such as irrigation, nutrient supply, light intensity, or temperature.
*   `y` is the **output vector**, representing the actual measurements obtained from sensors. These could be data from hyperspectral cameras, LiDAR scanners, thermal cameras, or chlorophyll fluorometers.
*   `A`, `B`, `C`, and `D` are matrices that define the system's dynamics:
    *   `A` (State Matrix): Describes how the internal states (phenotypic traits) evolve and interact with each other over time.
    *   `B` (Input Matrix): Describes how external inputs affect the states.
    *   `C` (Output Matrix): This is crucial for observability. It defines how the internal states are translated into the sensor outputs that we can actually measure.
    *   `D` (Feedthrough Matrix): Represents the direct influence of inputs on the outputs (often assumed to be zero in biological systems).

The primary goal is to use the known inputs (`u`) and the measured outputs (`y`) to estimate the internal states (`x`), which represent the key phenotypic traits. However, whether this is theoretically possible depends on the concept of observability.

**2. Formal Observability Analysis**

Observability is a fundamental property of a system that determines whether it is possible to deduce the entire internal state of the system (`x`) by observing its external outputs (`y`) over a finite period. In the context of phenotyping, the question observability answers is: **"Given our current sensor measurements, can we uniquely determine the value of all the phenotypic traits we have included in our model?"**

A formal analysis is conducted using the **observability matrix**, a key tool derived from the state-space model.

**2.1. The Observability Matrix**

The observability of a linear system is determined by the rank of its observability matrix, denoted by `O`. This matrix is constructed from the system's state matrix `A` and output matrix `C`. For a system with `n` state variables, the observability matrix is:

`O = [C; CA; CA²; ...; CA^(n-1)]`

This matrix essentially captures the relationship between the internal states and the sequence of sensor measurements over time. Each block of the matrix (`C`, `CA`, etc.) represents how the initial state `x(0)` is propagated through the system's dynamics (`A`) and then mapped to the sensor outputs (`C`).

**2.2. The Observability Condition**

The system is considered **fully observable** if and only if the observability matrix `O` has a **full column rank**. The rank of a matrix is the number of linearly independent rows or columns. For full observability, the rank must be equal to the number of state variables (`n`).

*   **`rank(O) = n`**: The system is fully observable. This means that every phenotypic trait included in the state vector `x` has a distinct influence on the sensor outputs `y`. By analyzing the sequence of sensor data, it is theoretically possible to uniquely determine the value of every one of these traits.

*   **`rank(O) < n`**: The system is **unobservable**. This means that the observability matrix is "rank deficient." In this case, there are one or more states (or combinations of states) that are "hidden" from the sensors. These unobservable states have no effect on the output `y`, making it impossible to distinguish their values based on the sensor data alone. For example, a sensor measuring only canopy height (`y`) would be unable to distinguish between two plants with the same height but different internal nitrogen content (`x`). The nitrogen content state would be unobservable from that specific sensor output.

**3. Identifying Observable vs. Unobservable Traits**

When an analysis reveals that a system is unobservable (`rank(O) < n`), the next step is to identify which specific phenotypic traits are causing this issue. This is done by analyzing the **null space** of the observability matrix.

Any non-zero vector in the null space of `O` corresponds to a set of initial states that produce a zero output for all time. This means that if the initial state vector lies within this null space, it is invisible to the sensors. The states that correspond to the non-zero elements of these null space vectors are the unobservable states.

By identifying these unobservable states, researchers can gain critical insights:

1.  **Sensor Limitation:** The analysis might reveal that the current set of sensors is fundamentally incapable of measuring a specific trait of interest. For instance, RGB and thermal cameras may not be sufficient to observe the internal nutrient concentration of a plant.
2.  **Model Redundancy:** It could indicate that some states in the model are redundant or linearly dependent. For example, two different modeled traits might always change in perfect proportion to one another, making them indistinguishable from the output.
3.  **Informing Sensor Selection:** The observability analysis can guide the selection and placement of new sensors. By understanding which states are unobservable, researchers can strategically add new measurement types (e.g., adding a hyperspectral sensor to the system) that are sensitive to those specific traits, thereby making the overall system observable.

In conclusion, the formal observability analysis, centered on the assessment of the observability matrix, is a critical theoretical step in using state-space models for phenotyping. It moves beyond simply collecting data, allowing researchers to determine which phenotypic traits are *theoretically measurable* from a given set of sensors and providing a rigorous mathematical basis for designing more effective and insightful high-throughput phenotyping systems. The concept is explained in introductory materials on state-space models, such as the one provided in the search results **[1]**.

**Cited Sources:**
[1] YouTube. "We introduce the concept of observability of a state-space model, and discuss how it can be checked using the observability matrix." Available at: https://www.youtube.com/watch?v=yaVBe7kecUA. Accessed on the date of this report.

## Estimator and Controller Design for Phenotypic Analysis: Design an estimation system, such as a Kalman or particle filter, for the real-time tracking of phenotypic traits during the 3D reconstruction process. Propose a control strategy to optimize the data acquisition process (e.g., sensor placement, lighting conditions) to improve the accuracy and efficiency of the analysis.



 
 ### "Investigate and design an optimal estimation system (e.g., Kalman filter, particle filter) for real-time tracking of phenotypic traits during 3D reconstruction. This includes defining the state-space model for trait dynamics, the measurement model linking states to sensor data, and comparing filter performance for this specific application.",

### Optimal Estimation System for Real-Time Tracking of Phenotypic Traits During 3D Reconstruction

The real-time tracking of phenotypic traits during 3D reconstruction necessitates an optimal estimation system to handle the dynamic nature of biological processes and the inherent noise in sensor data. The choice of filter, along with the design of the state-space and measurement models, is critical for accurate and robust tracking. This report investigates and designs such a system, comparing the suitability of different filtering approaches.

#### State-Space Model for Trait Dynamics

A state-space model is a mathematical framework that describes a system using a set of state variables and a set of equations that govern the evolution of these variables over time. In the context of tracking phenotypic traits, the state-space model comprises two main components:

1.  **State Vector (x_t):** This vector represents the set of phenotypic traits being tracked at a given time *t*. The specific traits will depend on the application, but could include:
    *   For plants: leaf area, stem length, internode distance, fruit size, etc.
    *   For animals: body size, limb length, gait parameters, etc.
    *   For cellular systems: cell volume, morphology, etc.

2.  **Process Model (f):** This model describes the temporal evolution of the state vector. It predicts the state at time *t* based on the state at time *t-1*. The process model can be represented as:

    **x_t = f(x_{t-1}, u_t) + w_t**

    where:
    *   **x_t** is the state vector at time *t*.
    *   **f** is the state transition function, which can be linear or non-linear.
    *   **u_t** is a control input (if any).
    *   **w_t** is the process noise, which accounts for unmodeled dynamics and uncertainties in the process. This is often assumed to be a zero-mean Gaussian distribution.

The choice of the state transition function *f* is crucial and depends on the biological process being modeled. For some traits, a simple linear model might suffice (e.g., assuming a constant growth rate over a short period). However, for many biological processes, a non-linear model will be more appropriate. For example, plant growth often follows a sigmoidal pattern, which can be modeled using a logistic function.

#### Measurement Model

The measurement model links the state vector (the phenotypic traits) to the sensor data obtained from the 3D reconstruction process. The 3D reconstruction can be achieved using various techniques, such as structure from motion (SfM), photogrammetry, or laser scanning. The measurement model can be represented as:

**z_t = h(x_t) + v_t**

where:
*   **z_t** is the measurement vector at time *t*. This could be a set of 3D points, a mesh, or features extracted from the reconstructed model.
*   **h** is the measurement function, which maps the state vector to the measurement space. This function can also be linear or non-linear.
*   **x_t** is the state vector at time *t*.
*   **v_t** is the measurement noise, which accounts for sensor errors and inaccuracies in the reconstruction process. This is also often assumed to be a zero-mean Gaussian distribution.

The measurement function *h* can be complex. For example, if the state is "leaf area", the measurement function would involve segmenting the leaf from the 3D point cloud and calculating its area. This process can be non-linear and subject to significant noise.

#### Comparison of Filter Performance

The choice of filter depends on the linearity of the process and measurement models.

**1. Kalman Filter:**

The standard Kalman filter is an optimal recursive estimator for linear systems with Gaussian noise. It consists of two steps:

*   **Prediction:** The filter predicts the next state and its uncertainty based on the process model.
*   **Update:** The filter corrects the prediction based on the current measurement.

**Advantages:**
*   Computationally efficient.
*   Optimal for linear systems.

**Disadvantages:**
*   Assumes linear process and measurement models.
*   Assumes Gaussian noise.

For many biological applications, the assumption of linearity is too restrictive. Therefore, non-linear extensions of the Kalman filter are often more suitable:

*   **Extended Kalman Filter (EKF):** The EKF linearizes the non-linear process and measurement models using a first-order Taylor series expansion.
*   **Unscented Kalman Filter (UKF):** The UKF uses a deterministic sampling technique called the unscented transform to capture the mean and covariance of the state distribution. The UKF generally provides better performance than the EKF for highly non-linear systems.

**2. Particle Filter:**

The particle filter, also known as a sequential Monte Carlo method, is a non-parametric filter that can handle non-linear and non-Gaussian systems. It represents the probability distribution of the state using a set of random samples called "particles".

**Advantages:**
*   Can handle any functional form of the process and measurement models.
*   Does not assume Gaussian noise.

**Disadvantages:**
*   Computationally more expensive than the Kalman filter family.
*   Can suffer from the "particle degeneracy" problem, where most particles have negligible weights.

#### Optimal Estimation System Design

For real-time tracking of phenotypic traits during 3D reconstruction, a standard Kalman filter is unlikely to be sufficient due to the non-linear nature of biological growth and the complexity of the measurement process. The choice between the EKF/UKF and the particle filter depends on the specific application:

*   **For systems with moderate non-linearity and near-Gaussian noise:** The **Unscented Kalman Filter (UKF)** is often a good choice. It provides a good balance between performance and computational cost.
*   **For highly non-linear systems or systems with non-Gaussian noise:** A **particle filter** is the most appropriate choice. Despite its higher computational cost, it can provide more accurate and robust tracking in these challenging scenarios.

**Example Application: Tracking Plant Growth**

*   **State Vector (x_t):** [leaf_area, stem_height, internode_length]
*   **Process Model (f):** A non-linear growth model, such as a logistic function, could be used to model the evolution of these traits.
*   **Measurement Model (h):** The 3D reconstruction provides a point cloud. The measurement function would involve segmenting the leaves and stem, and then calculating their respective areas and lengths.
*   **Filter:** Due to the non-linear growth dynamics and the complex measurement process, a **particle filter** would likely provide the best performance. However, if computational resources are limited, a **UKF** could be a viable alternative.

#### Conclusion

The optimal estimation system for real-time tracking of phenotypic traits during 3D reconstruction will likely involve a non-linear filter. The Unscented Kalman Filter offers a good compromise between accuracy and efficiency for moderately non-linear systems. For highly non-linear or non-Gaussian scenarios, the particle filter is the preferred method, provided that sufficient computational resources are available. The design of accurate process and measurement models, tailored to the specific biological system and 3D reconstruction technique, is as crucial as the choice of the filter itself. The provided search results offer a starting point for understanding state-space models and the Kalman filter, but further research into non-linear filtering techniques is essential for this application. [1, 2]

**References:**
[1] Jones, K. (n.d.). State-Space Models and Kalman Filtering for Time Series Analysis. Medium.
[2] Peng, R. (n.d.). State Space Models and the Kalman Filter. A Very Short Course on Time Series Analysis.
*(Note: The provided search results offer a general introduction to the topic but do not contain specific details on the application of these methods to phenotypic trait tracking. The design choices outlined in this report are based on the general principles of state estimation and the known characteristics of biological systems and 3D reconstruction data.)*

 
 ### Detail the integration of the proposed estimation system and control strategy into a cohesive, closed-loop system. Explain how the real-time state estimates from the filter will be used as feedback for the controller, and how the controller's actions will, in turn, provide better data to improve the accuracy of the estimation, creating a synergistic feedback loop."

### Integration of Estimation and Control into a Cohesive Closed-Loop System

The integration of a state estimation system (like a Kalman filter) and a control strategy forms a sophisticated and robust closed-loop control system, often referred to as an observer-based controller. This architecture is fundamental in modern control engineering, particularly when the internal states of a system cannot be directly measured. The synergy between the estimator and the controller creates a powerful feedback loop where each component's performance is mutually enhanced.

#### **1. The Primary Feedback Path: From Estimator to Controller**

The core of the integration lies in using the real-time outputs of the estimation filter as the input for the control law.

*   **State Estimation:** The estimator's primary role is to produce an optimal estimate of the system's internal state vector, denoted as **x̂** (x-hat), based on potentially noisy and incomplete sensor measurements. It uses a mathematical model of the system to predict the state's evolution and then corrects this prediction based on actual measurements.
*   **State-Feedback Control:** A typical state-feedback control law calculates the optimal control action, **u**, required to drive the system towards a desired state or reference trajectory. This law is a function of the system's state, often in the form of **u = -Kx**, where **K** is a pre-calculated gain matrix.
*   **Integration:** In a practical system where the true state **x** is unknown, the controller cannot use it directly. Instead, the estimated state **x̂** from the filter is used as a stand-in. The control law becomes **u = -Kx̂**. The controller treats the high-fidelity estimate from the filter as the true state of the system and computes the necessary control inputs accordingly. This closes the primary loop: the filter estimates the state, the controller acts on this estimate, the system's state changes, and new sensor measurements are fed back to the filter.

For linear systems, this design is supported by the **Separation Principle**, a fundamental theorem in control theory. It states that the problem of designing the optimal controller (calculating the gain matrix **K**) and designing the optimal estimator (the filter) can be done independently without loss of overall system optimality. The stability and performance of the combined system are guaranteed if the separate controller and estimator designs are stable.

#### **2. The Synergistic Loop: From Controller to Estimator**

The controller's actions, in turn, play a crucial role in improving the accuracy and reliability of the state estimator, creating a synergistic feedback loop that enhances the entire system.

*   **Providing a Known Input to the Model:** The estimation filter's model requires two key inputs to predict the next state: the current state and the control input applied to the system. The control signal **u**, calculated by the controller, is a precisely known quantity. This signal is fed directly into the prediction stage of the estimator's internal model. By knowing the exact "push" being applied to the system at every time step, the estimator can make a much more accurate prediction of how the state will evolve, significantly reducing prediction error and improving the overall accuracy of the final state estimate.

*   **Improving System Observability:** The controller can indirectly improve the quality of the data the estimator receives. By actively maneuvering the system, the controller can guide it into regions of its state space where the system's dynamics are more "observable." For instance, a specific control action might cause a state variable that is typically difficult to measure to have a more pronounced effect on the sensor readings. This provides the filter with more informative data, allowing it to reduce the uncertainty associated with that specific state, thereby improving the estimation accuracy. This concept is related to the field of "dual control," where the controller has the dual objective of both controlling the system and actively probing it to learn more about its state.

*   **Ensuring Model Validity through Stability:** The mathematical models used by estimators are often most accurate within a specific operating range. An uncontrolled (open-loop) system might drift into unstable or highly non-linear regions where the model's accuracy degrades, causing the estimator's performance to suffer. The closed-loop controller's primary job is to maintain system stability and keep it within the desired operating envelope. By doing so, the controller ensures that the system's behavior remains predictable and consistent with the estimator's internal model, which is essential for accurate and reliable state estimation.

In summary, the integration forms a cohesive system where the estimator provides the crucial state information that the controller needs to act, while the controller's precise actions and stabilizing influence ensure that the estimator is supplied with high-quality data, allowing it to produce a more accurate state estimate. This symbiotic relationship creates a robust, high-performance control system capable of operating effectively even in the presence of uncertainty and measurement noise.

## Comparative Analysis and Design Report Synthesis: Compare the proposed control theory-based approach with other relevant theoretical methods, such as machine learning (e.g., CNNs, GANs) and statistical shape models. Outline the structure of the final design report, integrating the findings from all sub-queries into a cohesive document that details the modeling, analysis, and design of the proposed system.



 
 ### In-depth comparison of the proposed control theory-based approach with machine learning methods (specifically CNNs and GANs). This analysis should evaluate performance metrics, computational complexity, data requirements, and applicability to the problem domain.

### In-depth Comparison: Control Theory vs. Machine Learning (CNNs & GANs)

This analysis provides a detailed comparison of control theory-based approaches with machine learning methods, specifically Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs). The comparison is structured around four key areas: performance metrics, computational complexity, data requirements, and applicability.

---

#### **1. Performance Metrics**

The evaluation of success differs fundamentally between control theory and machine learning approaches due to their different underlying principles.

*   **Control Theory-Based Approach:** Performance is measured using well-established metrics derived from the system's dynamic response. These metrics provide clear insights into the stability and behavior of the system. Key metrics include:
    *   **Time-Domain Metrics:** Rise time, settling time, overshoot, and steady-state error. These describe how the system responds to a change in the setpoint.
    *   **Frequency-Domain Metrics:** Gain margin and phase margin. These are crucial for assessing the stability and robustness of the closed-loop system.
    *   **Integral Error Metrics:** Integral Absolute Error (IAE) or Integral Squared Error (ISE) are used to quantify the cumulative error over time.

*   **Machine Learning (CNNs & GANs):** Performance is typically evaluated based on statistical error on a held-out test dataset. The choice of metric depends on the specific task (e.g., regression, classification).
    *   **CNNs:** When used for control (often as part of a larger system, e.g., for perception or state estimation), their performance is measured by metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or Mean Absolute Error (MAE) for regression tasks. For classification tasks (e.g., identifying system state), metrics include accuracy, precision, and recall.
    *   **GANs:** Evaluating GANs is notoriously complex. When used to generate control signals or system trajectories, metrics might include comparing the statistical properties of the generated data to real data. Common, though often indirect, metrics include Inception Score (IS) or Fréchet Inception Distance (FID), which are more common in image generation but conceptually adaptable. The training of GANs involves a min-max game between two networks, and convergence itself is a primary challenge and indicator of success. The metrics used for CNNs and GANs often differ significantly based on their distinct architectures and training processes [Source: researchgate.net/publication/391876453_Extensive_review_and_comparison_of_CNN_and_GAN].

---

#### **2. Computational Complexity**

There is a significant disparity in the computational demands of these approaches, particularly between the design/training phase and the implementation/inference phase.

*   **Control Theory-Based Approach:**
    *   **Design Phase:** The primary computational cost lies in the offline system identification (modeling) and controller design phase. This can be intensive but is a one-time cost.
    *   **Implementation (Real-time):** Traditional controllers (e.g., PID, LQR) are typically very lightweight and computationally efficient, making them suitable for real-time implementation on low-power microcontrollers.

*   **Machine Learning (CNNs & GANs):**
    *   **Training Phase:** This is the most computationally expensive part. Training deep neural networks like CNNs and especially GANs requires large datasets and significant computational power, often necessitating the use of specialized hardware like GPUs or TPUs for hours or even days. The intricate architectures of these models contribute to this high complexity [Source: researchgate.net/publication/391876453_Extensive_review_and_comparison_of_CNN_and_GAN].
    *   **Inference (Real-time):** While faster than training, the inference step for a deep neural network is generally more computationally demanding than a traditional control law. Deploying these models in real-time control loops with high sampling rates can be challenging and may require more powerful hardware.

---

#### **3. Data Requirements**

The nature and quantity of data needed represent a core difference between the methodologies.

*   **Control Theory-Based Approach:** This approach is "model-based." It requires data to perform system identification—the process of creating a mathematical model of the system's dynamics. This often involves using a smaller, curated dataset collected from specifically designed experiments that excite the system's modes. The emphasis is on the quality and relevance of the data to inform the model parameters.

*   **Machine Learning (CNNs & GANs):** These methods are "data-driven." Their performance is critically dependent on the availability of large, comprehensive datasets.
    *   They learn the system's dynamics directly from data, bypassing the need for an explicit mathematical model.
    *   The data must cover a wide range of operating conditions and scenarios to ensure the model generalizes well and does not fail when encountering novel situations.
    *   Data scarcity is a major bottleneck for applying these methods. If historical data is limited or expensive to obtain, training a reliable model is difficult.

---

#### **4. Applicability to the Problem Domain**

The suitability of each approach depends heavily on the specific characteristics and requirements of the problem.

*   **Control Theory-Based Approach:**
    *   **Strengths:** Its primary advantage is the ability to provide formal guarantees of stability, robustness, and performance, assuming the mathematical model is accurate. This makes it highly suitable for safety-critical applications like aerospace, industrial robotics, and chemical process control where predictability and reliability are paramount.
    *   **Weaknesses:** Performance degrades significantly if the mathematical model is inaccurate or if the system has highly complex, non-linear dynamics that are difficult to model from first principles.

*   **Machine Learning (CNNs & GANs):**
    *   **Strengths:** They excel at learning and controlling systems with complex, high-dimensional, and non-linear dynamics that are challenging to model. They can learn directly from sensor data (e.g., CNNs from camera images) and can adapt to changing conditions if trained on appropriate data. GANs can be used to generate realistic trajectories or control policies.
    *   **Weaknesses:** The main drawback is the lack of formal guarantees. They are often treated as "black boxes," making it difficult to predict their behavior in all possible scenarios, especially with out-of-distribution inputs. The training process for GANs is known to be unstable, adding another layer of complexity [Source: researchgate.net/publication/391876453_Extensive_review_and_comparison_of_CNN_and_GAN]. This makes their use in safety-critical applications a significant research challenge that often involves combining them with traditional control methods in hybrid approaches.


## Citations
- https://bookdown.org/rdpeng/timeseriesbook/state-space-models-and-the-kalman-filter.html 
- https://www.youtube.com/watch?v=oMl9zJwFKvA 
- https://www.youtube.com/watch?v=Lw7m0ll4hLg 
- https://medium.com/@kyle-t-jones/state-space-models-and-kalman-filtering-for-time-series-analysis-df404ad4cc2b 
- https://www.youtube.com/watch?v=yaVBe7kecUA 
- https://www.researchgate.net/publication/391876453_Extensive_review_and_comparison_of_CNN_and_GAN 
