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
#' Consider sample of n size, \eqn{y_1, \ldots, y_n}.
#' Let \eqn{s = n - p}.
#' \eqn{y_1, \ldots, y_n} data are rearranged as follows.
#' \deqn{Y_j = (y_j, y_{j + 1}, \ldots, y_{j + s - 1})^\intercal}
#' and \eqn{Z_j = (\epsilon_j, \epsilon_{j + 1}, \ldots, \epsilon_{j + s - 1})^\intercal}
#' For ordinary least squares (OLS) estimation,
#' we define each response matrix and design matrix in multivariate OLS as follows.
#' First, response matrix:
#' \deqn{Y_0 = Y_{p + 1}}
#' Next, design matrix:
#' \deqn{X_0 = [Y_p, \ldots, Y_1, 1]}
#' Then we now have OLS model
#' \deqn{Y_0 = X_0 A + Z}
#' where \eqn{Z = Z_{p + 1}}
#' Here,
#' \deqn{A = [A_1, A_2, \ldots, A_p, c]^T}
#' This gives that
#' \deqn{\hat{A} = (X_0^\intercal X_0)^{-1} X_0^\intercal Y_0}
#' 
#' # Vector Heterogeneous Autoregressive Model
#' 
#' * VHAR model is linearly restricted VAR(22).
#' * Let \eqn{Y_0 = X_1 \Phi + Z} be the OLS formula of VHAR.
#' * Let \eqn{T_0} be 3m x 22m matrix.
#' 
#' \deqn{
#'   T_0 = \begin{bmatrix}
#'   1 & 0 & 0 & 0 & 0 & 0 & \cdots & 0 \\
#'   1 / 5 & 1 / 5 & 1 / 5 & 1 / 5 & 1 / 5 & 0 & \cdots & 0 \\
#'   1 / 22 & 1 / 22 & 1 / 22 & 1 / 22 & 1 / 22 & 1 / 22 & \cdots & 1 / 22
#' \end{bmatrix} \otimes I_m
#' }
#' 
#' Define (3m + 1) x (22m + 1) matrix \eqn{T_{HAR}} by
#' \deqn{
#'   T_{HAR} \defn \begin{bmatrix}
#'   T_0 & 0_{3m} \\
#'   0_{3m}^\intercal & 1
#' \end{bmatrix}
#' }
#' 
#' Then by construction,
#' \deqn{Y_0 = X_1 \Phi + Z = (X_0 T_{HAR}^\intercal) \Phi + Z}
#' i.e.
#' \deqn{X_1 = X_0 T_{HAR}^\intercal}
#' 
#' It follows that
#' \deqn{\hat\Phi = (X_1^\intercal X_1)^{-1} X_1^\intercal Y_0}
#' 
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
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
#' J_p \otimes diag\left( \sigma_1, \ldots, \sigma_m \right) / \lambda & \zero_{mp} \\ \hline
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
#'   J_3 \otimes diag\left( \sigma_1, \ldots, \sigma_m \right) / \lambda & \zero_{3m} \\ \hline
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
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
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
#' 
#' @keywords internal
#' @name bvar_predictive_density
NULL
