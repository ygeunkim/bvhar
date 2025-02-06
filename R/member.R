#' Coefficient Matrix of Multivariate Time Series Models
#' 
#' By defining [stats::coef()] for each model, this function returns coefficient matrix estimates.
#' 
#' @param object Model object
#' @param ... not used
#' @return [matrix] object with appropriate dimension.
#' @name coef
#' @export
coef.varlse <- function(object, ...) {
  object$coefficients
}

#' @rdname coef
#' @export
coef.vharlse <- function(object, ...) {
  object$coefficients
}

#' @rdname coef
#' @export
coef.bvarmn <- function(object, ...) {
  object$coefficients
}

#' @rdname coef
#' @export
coef.bvarflat <- function(object, ...) {
  object$coefficients
}

#' @rdname coef
#' @export
coef.bvharmn <- function(object, ...) {
  object$coefficients
}

#' @rdname coef
#' @export
coef.bvharsp <- function(object, ...) {
  object$coefficients
}

#' @rdname coef
#' @export
coef.summary.bvharsp <- function(object, ...) {
  object$coefficients
}

#' Residual Matrix from Multivariate Time Series Models
#' 
#' By defining [stats::residuals()] for each model, this function returns residual.
#' 
#' @param object Model object
#' @param ... not used
#' @return [matrix] object.
#' @name residuals
#' @export
residuals.varlse <- function(object, ...) {
  object$residuals
}

#' @rdname residuals
#' @export
residuals.vharlse <- function(object, ...) {
  object$residuals
}

#' @rdname residuals
#' @export
residuals.bvarmn <- function(object, ...) {
  object$residuals
}

#' @rdname residuals
#' @export
residuals.bvarflat <- function(object, ...) {
  object$residuals
}

#' @rdname residuals
#' @export
residuals.bvharmn <- function(object, ...) {
  object$residuals
}

#' Fitted Matrix from Multivariate Time Series Models
#' 
#' By defining [stats::fitted()] for each model, this function returns fitted matrix.
#' 
#' @param object Model object
#' @param ... not used
#' @return [matrix] object.
#' @name fitted
#' @export
fitted.varlse <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted
#' @export
fitted.vharlse <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted
#' @export
fitted.bvarmn <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted
#' @export
fitted.bvarflat <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted
#' @export
fitted.bvharmn <- function(object, ...) {
  object$fitted.values
}

#' @rdname var_lm
#' @param x Any object
#' @export
is.varlse <- function(x) {
  inherits(x, "varlse")
}

#' @rdname vhar_lm
#' @param x Any object
#' @export
is.vharlse <- function(x) {
  inherits(x, "vharlse")
}

#' @rdname bvar_minnesota
#' @param x Any object
#' @export
is.bvarmn <- function(x) {
  inherits(x, "bvarmn")
}

#' @rdname bvar_flat
#' @param x Any object
#' @export
is.bvarflat <- function(x) {
  inherits(x, "bvarflat")
}

#' @rdname bvhar_minnesota
#' @param x Any object
#' @export
is.bvharmn <- function(x) {
  inherits(x, "bvharmn")
}

#' @rdname var_lm
#' @export
is.bvharmod <- function(x) {
  inherits(x, "bvharmod")
}

#' @rdname predict
#' @param x Any object
#' @export
is.predbvhar <- function(x) {
  inherits(x, "predbvhar")
}

#' @rdname forecast_roll
#' @param x Any object
#' @export
is.bvharcv <- function(x) {
  inherits(x, "bvharcv")
}

#' @rdname irf
#' @param x Any object
#' @export
is.bvharirf <- function(x) {
  inherits(x, "bvharirf")
}

#' @rdname set_bvar
#' @param x Any object
#' @export
is.bvharspec <- function(x) {
  inherits(x, "bvharspec")
}

#' @rdname set_lambda
#' @param x Any object
#' @export
is.bvharpriorspec <- function(x) {
  inherits(x, "bvharpriorspec")
}

#' @rdname choose_bvar
#' @param x Any object
#' @export
is.bvharemp <- function(x) {
  inherits(x, "bvharemp")
}

#' @rdname bound_bvhar
#' @param x Any object
#' @export
is.boundbvharemp <- function(x) {
  inherits(x, "boundbvharemp")
}

#' @rdname set_intercept
#' @param x Any object
#' @export
is.interceptspec <- function(x) {
  inherits(x, "interceptspec")
}

#' @rdname set_ssvs
#' @param x Any object
#' @export
is.ssvsinput <- function(x) {
  inherits(x, "ssvsinput")
}

#' @rdname set_horseshoe
#' @param x Any object
#' @export
is.horseshoespec <- function(x) {
  inherits(x, "horseshoespec")
}

#' @rdname set_ng
#' @param x Any object
#' @export
is.ngspec <- function(x) {
  inherits(x, "ngspec")
}

#' @rdname set_dl
#' @param x Any object
#' @export
is.dlspec <- function(x) {
  inherits(x, "dlspec")
}

#' @rdname set_gdp
#' @param x Any object
#' @export
is.gdpspec <- function(x) {
  inherits(x, "gdpspec")
}

#' @rdname set_ldlt
#' @param x Any object
#' @export
is.covspec <- function(x) {
  inherits(x, "covspec")
}

#' @rdname set_ldlt
#' @param x Any object
#' @export
is.svspec <- function(x) {
  inherits(x, "svspec")
}

#' @rdname set_ldlt
#' @param x Any object
#' @export
is.ldltspec <- function(x) {
  inherits(x, "ldltspec")
}
