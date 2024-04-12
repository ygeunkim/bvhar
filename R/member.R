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
#' @param x `varlse` object
#' @export
is.varlse <- function(x) {
  inherits(x, "varlse")
}

#' @rdname vhar_lm
#' @param x `vharlse` object
#' @export
is.vharlse <- function(x) {
  inherits(x, "vharlse")
}

#' @rdname bvar_minnesota
#' @export
is.bvarmn <- function(x) {
  inherits(x, "bvarmn")
}

#' @rdname bvar_flat
#' @export
is.bvarflat <- function(x) {
  inherits(x, "bvarflat")
}

#' @rdname bvhar_minnesota
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
#' @export
is.predbvhar <- function(x) {
  inherits(x, "predbvhar")
}

#' @rdname forecast_roll
#' @export
is.bvharcv <- function(x) {
  inherits(x, "bvharcv")
}

#' @rdname irf
#' @export
is.bvharirf <- function(x) {
  inherits(x, "bvharirf")
}

#' @rdname set_bvar
#' @export
is.bvharspec <- function(x) {
  inherits(x, "bvharspec")
}

#' @rdname set_lambda
#' @export
is.bvharpriorspec <- function(x) {
  inherits(x, "bvharpriorspec")
}

#' @rdname choose_bvar
#' @export
is.bvharemp <- function(x) {
  inherits(x, "bvharemp")
}

#' @rdname bound_bvhar
#' @export
is.boundbvharemp <- function(x) {
  inherits(x, "boundbvharemp")
}

#' @rdname set_intercept
#' @export
is.interceptspec <- function(x) {
  inherits(x, "interceptspec")
}

#' @rdname set_ssvs
#' @export
is.ssvsinput <- function(x) {
  inherits(x, "ssvsinput")
}

#' @rdname init_ssvs
#' @export
is.ssvsinit <- function(x) {
  inherits(x, "ssvsinit")
}

#' @rdname set_horseshoe
#' @export
is.horseshoespec <- function(x) {
  inherits(x, "horseshoespec")
}

#' @rdname set_sv
#' @export
is.svspec <- function(x) {
  inherits(x, "svspec")
}
