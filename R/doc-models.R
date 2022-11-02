#' Ordinary Least Squares Model Formulation
#' 
#' @description 
#' This page specifies the formulation for ordinary least squares (OLS) that VAR-family model.
#' Notation and format here will be used in this entire package document.
#' 
#' # Vector Autoregressive Model
#' 
#' As mentioned in [var_lm()], we specify VAR(p) model by
#' \deqn{Y_t = A_1 Y_{t - 1} + \cdots + A_p Y_{t - p} + c + \epsilon_t}
#' Consider sample of T size, \eqn{y_1, \ldots, y_n}.
#' Let \eqn{n = T - p}.
#' \eqn{y_1, \ldots, y_T} data are rearranged as follows.
#' \deqn{Y_j = (y_j, y_{j + 1}, \ldots, y_{j + n - 1})^\intercal}
#' and \eqn{Z_j = (\epsilon_j, \epsilon_{j + 1}, \ldots, \epsilon_{j + n - 1})^\intercal}
#' For ordinary least squares (OLS) estimation,
#' we define each response matrix and design matrix in multivariate OLS as follows.
#' First, response matrix:
#' \deqn{Y_0 = Y_{p + 1}}
#' Next, design matrix:
#' \deqn{X_0 = [Y_p, \ldots, Y_1, 1]}
#' Then we now have OLS model
#' \deqn{Y_0 = X_0 A + Z_0}
#' where \eqn{Z_0 = Z_{p + 1}}
#' Here,
#' \deqn{A = [A_1, A_2, \ldots, A_p, c]^T}
#' This gives that
#' \deqn{\hat{A} = (X_0^\intercal X_0)^{-1} X_0^\intercal Y_0}
#' 
#' # Vector Heterogeneous Autoregressive Model
#' 
#' * VHAR model is linearly restricted VAR(22).
#' * Let \eqn{Y_0 = X_1 \Phi + Z_0} be the OLS formula of VHAR.
#' * Let \eqn{T_0} be 3m x 22m matrix.
#' 
#' \deqn{
#'   C_0 = \begin{bmatrix}
#'   1 & 0 & 0 & 0 & 0 & 0 & \cdots & 0 \\
#'   1 / 5 & 1 / 5 & 1 / 5 & 1 / 5 & 1 / 5 & 0 & \cdots & 0 \\
#'   1 / 22 & 1 / 22 & 1 / 22 & 1 / 22 & 1 / 22 & 1 / 22 & \cdots & 1 / 22
#' \end{bmatrix} \otimes I_m
#' }
#' 
#' Define (3m + 1) x (22m + 1) matrix \eqn{T_{HAR}} by
#' \deqn{
#'   C_{HAR} \defn \begin{bmatrix}
#'   C_0 & 0_{3m} \\
#'   0_{3m}^\intercal & 1
#' \end{bmatrix}
#' }
#' 
#' Then by construction,
#' \deqn{Y_0 = X_1 \Phi + Z_0 = (X_0 C_{HAR}^\intercal) \Phi + Z_0}
#' i.e.
#' \deqn{X_1 = X_0 C_{HAR}^\intercal}
#' 
#' It follows that
#' \deqn{\hat\Phi = (X_1^\intercal X_1)^{-1} X_1^\intercal Y_0}
#' 
#' @references 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. doi:[10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
#' 
#' Bubák, V., Kočenda, E., & Žikeš, F. (2011). *Volatility transmission in emerging European foreign exchange markets*. Journal of Banking & Finance, 35(11), 2829–2841. doi:[10.1016/j.jbankfin.2011.03.012](https://doi.org/10.1016/j.jbankfin.2011.03.012)
#' 
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196. doi:[10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001)
#' 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' @keywords internal
#' @name var_design_formulation
NULL

#' Adding Dummy Observations
#' 
#' @description 
#' This page notes how to define dummy observation matrices in Bayesian VAR and VHAR models.
#' 
#' # Bayesian VAR
#' 
#' Consider BVAR and its hyperparameters in [set_bvar()].
#' * `sigma`: \eqn{\sigma_1, \ldots, \sigma_m}
#' * `lambda`: \eqn{\lambda}
#' * `delta`: \eqn{\delta_1, \ldots, \delta_m}
#' * `eps`: \eqn{\epsilon}
#' 
#' This package implements adding-dummy-observations approach of Bańbura et al. (2010).
#' Let \eqn{J_p = diag(1, 2, \ldots, p)}.
#' For each response and design matrix \eqn{Y_0} and \eqn{X_0}, we define dummy observation named by \eqn{Y_p} and \eqn{X_p}.
#' Response dummy matrix \eqn{Y_p} is (m + k) x m matrix:
#' \deqn{
#'   Y_p = \left[\begin{array}{c}
#' diag\left( \delta_1 \sigma_1, \ldots, \delta_m \sigma_m \right) / \lambda \\
#' 0_{m(p - 1) \times m} \\ \hline
#' diag\left( \sigma_1, \ldots, \sigma_m \right) \\ \hline
#' 0_m^\intercal
#' \end{array}\right]
#' }
#' Design dummy matrix \eqn{X_p} is (m + k) x k matrix:
#' \deqn{
#'   X_p = \left[\begin{array}{c|c}
#' J_p \otimes diag\left( \sigma_1, \ldots, \sigma_m \right) / \lambda & 0_{mp} \\ \hline
#' 0_{m \times mp} & 0_m \\ \hline
#' 0_{mp}^\intercal & \epsilon
#' \end{array}\right]
#' }
#' 
#' These two matrices define Minnesota prior distribution of BVAR.
#' 
#' # Bayesian VHAR
#' 
#' Consider BVHAR and its hyperparameter in [set_bvhar()].
#' First, VAR-type minnesota prior:
#' * `sigma`: \eqn{\sigma_1, \ldots, \sigma_m}
#' * `lambda`: \eqn{\lambda}
#' * `delta`: \eqn{\delta_1, \ldots, \delta_m}
#' * `eps`: \eqn{\epsilon}
#' For response matrix \eqn{Y_0}, define (m + h) x m matrix \eqn{Y_{HAR}}
#' \deqn{
#'   Y_{HAR} = \left[\begin{array}{c}
#'   diag\left( \delta_1 \sigma_1, \ldots, \delta_m \sigma_m \right) / \lambda \\
#'   0_{2m \times m} \\ \hline
#'   diag\left( \sigma_1, \ldots, \sigma_m \right) \\ \hline
#'   0_m^\intercal
#' \end{array}\right]
#' }
#' For design matrix \eqn{X_0}, define (m + h) x h matrix \eqn{X_{HAR}}
#' \deqn{
#'   X_{HAR} = \left[\begin{array}{c|c}
#'   J_3 \otimes diag\left( \sigma_1, \ldots, \sigma_m \right) / \lambda & 0_{3m} \\ \hline
#'   0_{m \times 3m} & 0_m \\ \hline
#'   0_{3m}^\intercal & \epsilon
#' \end{array}\right]
#' }
#' In case of VHAR-type minnesota prior, `delta` is replaced with the following three:
#' * `daily`: \eqn{d_1, \ldots, d_m}
#' * `weekly`: \eqn{w_1, \ldots, w_m}
#' * `monthly`: \eqn{m_1, \ldots, m_m}
#' and \eqn{Y_{HAR}} is changed.
#' \deqn{
#'   Y_{HAR} = \left[\begin{array}{c}
#'   diag\left( d_1 \sigma_1, \ldots, d_m \sigma_m \right) / \lambda \\
#'   diag\left( w_1 \sigma_1, \ldots, w_m \sigma_m \right) / \lambda \\
#'   diag\left( m_1 \sigma_1, \ldots, m_m \sigma_m \right) / \lambda \\ \hline
#'   diag\left( \sigma_1, \ldots, \sigma_m \right) \\ \hline
#'   0_m^\intercal
#' \end{array}\right]
#' }
#' 
#' These two dummy matrices define minnesota prior distribution of BVHAR.
#' 
#' @references Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). doi:[10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' @keywords internal
#' @name bvar_adding_dummy
NULL

#' Time Series Cross-Validation
#' 
#' @description 
#' This page describes the out-of-sample forecasting method in time series scheme.
#' While the most simple way to compute test error is splitting training-test set,
#' but popular approach in time series literature is out-of-sample forecasting.
#' 
#' # Rolling Window Forecasting
#' 
#' Rolling window forecasting fixes its window size.
#' The window is used as training set.
#' This window will be moved to the end as possible as it can be.
#' 
#' The step should set the same step ahead forecasting at every iteration, saying one-step or h-step.
#' After fitting the model in the window, researcher should forecast next one-step or h-step ahead.
#' The longest forecast horizon is `(final_period - window_size) - h + 1`.
#' 
#' After this window, move the window one step ahead and do the same process.
#' Get forecasted values until possible (longest forecast horizon).
#' 
#' # Expanding Windows
#' 
#' Expanding window forecasting fixes its starting period.
#' Rolling window method moves the window with constant size,
#' but expanding window method just literally expands window the window fixing the starting period.
#' 
#' The other procedure is the same.
#' 
#' @references 
#' Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTEXTS. [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
#' 
#' @keywords internal
#' @name ts_forecasting_cv
NULL

#' Predictive Density of Bayesian Models
#' 
#' @description 
#' This page explains the simulation algorithm for predictive distribution of BVAR and BVHAR.
#' 
#' # Simulating predictive distribution of BVAR
#' 
#' This simulation process is required because we do not know the closed form of h-step ahead forecasting density.
#' For given number of simulation (`n_iter`),
#' 
#' 1. Generate \eqn{(A^{(b)}, \Sigma_e^{(b)}) \sim MIW} (posterior)
#' 2. Recursively, \eqn{j = 1, \ldots, h} (`n_ahead`)
#'     - Point forecast: Use \eqn{\hat{A}}
#'     - Predictive distribution: Again generate \eqn{\tilde{Y}_{n + j}^{(b)} \sim A^{(b)}, \Sigma_e^{(b)} \sim MN}
#'     - tilde notation indicates simulated ones
#' 
#' Simulating predictive distribution of BVHAR
#' 
#' We extend the similar procedure in BVAR to the BVHAR.
#' 
#' For given number of simulation (`n_iter`),
#' 
#' 1. Generate \eqn{(\Phi^{(b)}, \Sigma_e^{(b)}) \sim MIW} (posterior)
#' 2. Recursively, \eqn{j = 1, \ldots, h} (`n_ahead`)
#'     - Point forecast: Use \eqn{\hat\Phi}
#'     - Predictive distribution: Again generate \eqn{\tilde{Y}_{n + j}^{(b)} \sim \Phi^{(b)}, \Sigma_e^{(b)} \sim MN}
#'     - tilde notation indicates simulated ones
#' 
#' @references 
#' Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2). doi:[10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
#' 
#' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791–897. doi:[10.1016/b978-0-444-62731-5.00015-4](https://doi.org/10.1016/B978-0-444-62731-5.00015-4)
#' @keywords internal
#' @name bvar_predictive_density
NULL

#' Vectorized OLS Formulation
#' 
#' @description 
#' This page specifies the OLS formulation, which is vectorized of [var_design_formulation].
#' Notation and format here will be used in this entire package document.
#' 
#' # Vector Autoregressive Model
#' 
#' Recall k-dim VAR model \eqn{Y_0 = X_0 A + Z_0}.
#' Applying \eqn{vec} operation, we have
#' 
#' \deqn{vec(Y_0) = (I_k \otimes X_0) vec(A) + vec(Z_0)}
#' 
#' Estimating \eqn{\alpha = vec(A)} is equivalent to estimating usual OLS.
#' 
#' # Vector Heterogeneous Autoregressive Model
#' 
#' Recall k-dim VHAR model \deqn{Y_0 = X_1 \Phi + Z_0 = (X_0 C_{HAR}^\intercal) \Phi + Z_0}.
#' Then
#' \deqn{vec(Y_0) = (I_k \otimes X_0 C_{HAR}^\intercal) vec(\Phi) + vec(Z_0) = (I_k \otimes X_1) vec(\Phi) + vec(Z_0)}
#' 
#' # Restrictions
#' 
#' \eqn{\phi = vec(\Phi)} is linear restriction of \eqn{\alpha = vec(A)}.
#' Then
#' \deqn{\alpha = vec(C_{HAR}^\intercal \Phi) = (I_k \otimes C_{HAR}^\intercal) \phi}
#' 
#' # Restrictions for Sparsity
#' 
#' Value of 0, 1 corresponding to \eqn{A_{i,j}} gives sparsity of the element.
#' Let \eqn{\psi = (\psi_1, \ldots, \psi_{k^2 p})^\intercal} where \eqn{\psi_j = 0,1},
#' and let \eqn{\Psi = diag(\psi) \in \mathbb{R}^{k^2 p \times k^2 p}}.
#' 
#' Then the new VAR coefficient vector \eqn{\theta = \Psi \alpha \in \mathbb{R}^{k^2 p}} becomes
#' \eqn{\alpha_j = 0} if \eqn{\psi_j = 0} and \eqn{\alpha_j \neq 0} if \eqn{\psi = 1}.
#' 
#' 
#' 
#' @references 
#' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://doi.org/10.1016/j.ijforecast.2009.11.002)
#' 
#' Korobilis, D. (2013). *VAR FORECASTING USING BAYESIAN VARIABLE SELECTION*. Journal of Applied Econometrics, 28(2). doi:[10.1002/jae.1271](https://doi.org/10.1002/jae.1271)
#' 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @keywords internal
#' @name var_vec_formulation
NULL

#' Stochastic Search Variable Selection in VAR
#' 
#' @description 
#' This page describes a stochastic search variable selection (SSVS) MCMC algorithm
#' in a VAR model.
#' 
#' # SSVS Prior
#' 
#' Let \eqn{(\gamma_1, \ldots, \gamma_k)^\intercal} be dummy variables restricting coefficients vector.
#' Then
#' \deqn{
#'   h_i = \begin{cases}
#'     \tau_{0i} & \text{if } \gamma_i = 0 \\
#'     \tau_{1i} & \text{if } \gamma_i = 1
#'   \end{cases}
#' }
#' In turn, \eqn{D = diag(h_1, \ldots, h_k)}.
#' Let \eqn{\omega_j = (\omega_{1j}, \ldots, \omega_{j - 1, j})^\intercal} be dummy variables restricting covariance matrix.
#' Then
#' \deqn{
#'   h_{ij} = \begin{cases}
#'     \kappa_{0ij} & \text{if } \omega_{ij} = 0 \\
#'     \kappa_{1ij} & \text{if } \omega_{ij} = 1
#'   \end{cases}
#' }
#' In turn, \eqn{D_j = diag(h_{1j}, \ldots, h_{j-1, j})}.
#' 
#' 
#' After sampling \eqn{\psi_{jj}, \psi_{ij}}, and \eqn{\omega_{ij}}, generate \eqn{\alpha} by
#' combining non-restricted constant term and potentially restricted coefficients vector term.
#' This process is done outside of the function and gives each prior mean and prior precision.
#' In turn,
#' \deqn{\alpha \mid \gamma, \eta, \omega, \psi, Y_0 \sim N_{k^2 p} (\mu, \Delta)}
#' The dimension \eqn{k^2 p} is when non-constant term.
#' When there is constant term, it is \eqn{k (kp + 1)}.
#' Here,
#' \deqn{
#'   \mu = (
#''     (\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) + M^{-1}
#'   )^{-1} (
#''     ( (\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) ) \hat{\alpha}^{MLE} + M^{-1} \alpha_0
#'   )
#' }
#' where \eqn{\alpha_0} is the prior mean for \eqn{\alpha}.
#' In regression, MLE is the same as OLS.
#' \deqn{
#'   \Delta = ((\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) + M^{-1})^{-1}
#' }
#' After this step, we move to generating Bernoulli \eqn{\gamma_j}.
#' 
#' 
#' We draw \eqn{\omega_{ij}} and \eqn{\gamma_j} from Bernoulli distribution.
#' \deqn{
#'   \omega_{ij} \mid \eta_j, \psi_j, \alpha, \gamma, \omega_{j}^{(previous)} \sim Bernoulli(\frac{u_{ij1}}{u_{ij1} + u_{ij2}})
#' }
#' If \eqn{R_j = I_{j - 1}},
#' \deqn{
#'   u_{ij1} = \frac{1}{\kappa_{1ij} \exp(- \frac{\psi_{ij}^2}{2 \kappa_{1ij}^2}) q_{ij},
#'   u_{ij2} = \frac{1}{\kappa_{0ij} \exp(- \frac{\psi_{ij}^2}{2 \kappa_{0ij}^2}) (1 - q_{ij})
#' }
#' Otherwise, see George et al. (2008).
#'' Also,
#' \deqn{
#'   \gamma_j \mid \alpha, \psi, \eta, \omega, Y_0 \sim Bernoulli(\frac{u_{i1}}{u_{j1} + u_{j2}})
#' }
#' Similarly, if \eqn{R = I_{k^2 p}},
#' \deqn{
#'   u_{j1} = \frac{1}{\tau_{1j}} \exp(- \frac{\alpha_j^2}{2 \tau_{1j}^2})p_i,
#'   u_{j2} = \frac{1}{\tau_{0j}} \exp(- \frac{\alpha_j^2}{2 \tau_{0j}^2})(1 - p_i)
#' }
#' 
#' 
#' Let SSE matrix be \eqn{S(\hat{A}) = (Y_0 - X_0 \hat{A})^\intercal (Y_0 - X_0 \hat{A}) \in \mathbb{R}^{k \times k}},
#' let \eqn{S_j} be the upper-left j x j block matrix of \eqn{S(\hat{A})},
#' and let \eqn{s_j = (s_{1j}, \ldots, s_{j - 1, j})^\intercal}.
#' For specified shape and rate of Gamma distribuion \eqn{a_j} and \eqn{b_j},
#' \deqn{
#'   \psi_{jj}^2 \mid \alpha, \gamma, \omega, Y_0 \sim \gamma(a_i + n / 2, B_i)
#' }
#' where
#' \deqn{
#'   B_i = \begin{cases}
#'     b_1 + s_{11} / 2 & \text{if } i = 1 \\
#'     b_i + (s_{ii} - s_i^\intercal ( S_{i - 1} + (D_i R_i D_i)^(-1) )^(-1) s_i) & \text{if } i = 2, \ldots, k
#'   \end{cases}
#' }
#' , and \eqn{D_i = diag(h_{1j}, \ldots, h_{i - 1, i}) \in \mathbb{R}^{(j - 1) \times (j - 1)}}
#' is the one made by upper diagonal element of \eqn{\Psi} matrix.
#' 
#' 
#' After drawing \eqn{\psi_{jj}}, generate upper elements by
#' \deqn{
#'   \eta_j \mid \alpha, \gamma, \omega, \psi, Y_0 \sim N_{j - 1} (\mu_j, \Delta_j)
#' }
#' where
#' \deqn{
#'   \mu_j = -\psi_{jj} (S_{j - 1} + (D_j R_j D_j)^{-1})^{-1} s_j,
#'   \Delta_j = (S_{j - 1} + (D_j R_j D_j)^{-1})^{-1}
#' }
#' 
#' 
#' 
#' Consider \eqn{\Sigma_e^{-1} = \Psi \Psi^\intercal} where upper triangular \eqn{\Psi = [\psi_{ij}]}.
#' Column vector for upper off-diagonal element is denoted by
#' \deqn{
#'   \eta_j = (\psi_{12}, \ldots, \psi_{j-1, j})
#' }
#' for \eqn{j = 2, \ldots, k}.
#' 
#' # Gibbs Sampling
#' 
#' Data: \eqn{X_0}, \eqn{Y_0}
#' 
#' Input:
#' * VAR order p
#' * MCMC iteration number
#' * Weight of each slab: Bernoulli distribution parameters
#'     * \eqn{p_j}: of coefficients
#'     * \eqn{q_{ij}}: of cholesky factor
#' * Gamma distribution parameters for cholesky factor diagonal elements \eqn{\psi_{jj}}
#'     * \eqn{a_j}: shape
#'     * \eqn{b_j}: rate
#' * Correlation matrix of coefficient vector: \eqn{R = I_{k^2p}}
#' * Correlation matrix to restrict cholesky factor (of \eqn{\eta_j}): \eqn{R_j = I_{j - 1}}
#' * Tuning parameters for spike-and-slab sd semi-automatic approach
#'     * \eqn{c_0}: small value (0.1)
#'     * \eqn{c_1}: large value (10)
#' * Constant to reduce prior influence on constant term: \eqn{c}
#' 
#' Gibbs sampling:
#' 1. Initialize \eqn{\Psi}, \eqn{\omega}, \eqn{\alpha}, \eqn{\gamma}
#' 2. Iterate
#'     1. Diagonal elements of cholesky factor: \eqn{\psi^{(t)} \mid \alpha^{(t - 1)}, \gamma^{(t - 1)}, \omega^{(t - 1)}, Y_0}
#'     2. Off-diagonal elements of cholesky factor: \eqn{\eta^{(t)} \mid \psi^{(t)} \alpha^{(t - 1)}, \gamma^{(t - 1)}, \omega^{(t - 1)}, Y_0}
#'     3. Dummy vector for cholesky factor: \eqn{\omega^{(t)} \mid \eta^{(t)}, \psi^{(t)} \alpha^{(t - 1)}, \gamma^{(t - 1)}, \omega^{(t - 1)}, Y_0}
#'     4. Coefficient vector: \eqn{\alpha^{(t)} \mid \gamma^{(t - 1)}, \Sigma^{(t)}, \omega^{(t)}, Y_0}
#'     5. Dummy vector for coefficient vector: \eqn{\gamma^{(t)} \mid \alpha^{(t)}, \psi^{(t)}, \eta^{(t)}, \omega^{(t)}, Y_0}
#' 
#' 
#' @keywords internal
#' @name ssvs_bvar_algo
NULL
