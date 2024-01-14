#' Coefficient Matrix of Multivariate Time Series Models
#' 
#' By defining [stats::coef()] for each model, this function returns coefficient matrix estimates.
#' 
#' @param object Model object
#' @param ... not used
#' @return [matrix] object with appropriate dimension.
#' @export
coef.varlse <- function(object, ...) {
  object$coefficients
}

#' @rdname coef.varlse
#' @export
coef.vharlse <- function(object, ...) {
  object$coefficients
}

#' @rdname coef.varlse
#' @export
coef.bvarmn <- function(object, ...) {
  object$coefficients
}

#' @rdname coef.varlse
#' @export
coef.bvarflat <- function(object, ...) {
  object$coefficients
}

#' @rdname coef.varlse
#' @export
coef.bvharmn <- function(object, ...) {
  object$coefficients
}

#' @rdname coef.varlse
#' @export
coef.bvharsp <- function(object, ...) {
  object$coefficients
}

#' @rdname coef.varlse
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
#' @export
residuals.varlse <- function(object, ...) {
  object$residuals
}

#' @rdname residuals.varlse
#' @export
residuals.vharlse <- function(object, ...) {
  object$residuals
}

#' @rdname residuals.varlse
#' @export
residuals.bvarmn <- function(object, ...) {
  object$residuals
}

#' @rdname residuals.varlse
#' @export
residuals.bvarflat <- function(object, ...) {
  object$residuals
}

#' @rdname residuals.varlse
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
#' @export
fitted.varlse <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted.varlse
#' @export
fitted.vharlse <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted.varlse
#' @export
fitted.bvarmn <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted.varlse
#' @export
fitted.bvarflat <- function(object, ...) {
  object$fitted.values
}

#' @rdname fitted.varlse
#' @export
fitted.bvharmn <- function(object, ...) {
  object$fitted.values
}

#' See if the Object a class in this package
#' 
#' This function returns `TRUE` if the input is the [class] defined by this package.
#' 
#' @param x Object
#' 
#' @return logical class
#' 
#' @export
is.varlse <- function(x) {
  inherits(x, "varlse")
}

#' @rdname is.varlse
#' @export
is.vharlse <- function(x) {
  inherits(x, "vharlse")
}

#' @rdname is.varlse
#' @export
is.bvarmn <- function(x) {
  inherits(x, "bvarmn")
}

#' @rdname is.varlse
#' @export
is.bvarflat <- function(x) {
  inherits(x, "bvarflat")
}

#' @rdname is.varlse
#' @export
is.bvharmn <- function(x) {
  inherits(x, "bvharmn")
}

#' @rdname is.varlse
#' @export
is.predbvhar <- function(x) {
  inherits(x, "predbvhar")
}

#' @rdname is.varlse
#' @export
is.bvharcv <- function(x) {
  inherits(x, "bvharcv")
}

#' @rdname is.varlse
#' @export
is.bvharspec <- function(x) {
  inherits(x, "bvharspec")
}

#' @rdname is.varlse
#' @export
is.bvharpriorspec <- function(x) {
  inherits(x, "bvharpriorspec")
}

#' @rdname is.varlse
#' @export
is.bvharemp <- function(x) {
  inherits(x, "bvharemp")
}

#' @rdname is.varlse
#' @export
is.boundbvharemp <- function(x) {
  inherits(x, "boundbvharemp")
}

#' @rdname is.varlse
#' @export
is.interceptspec <- function(x) {
  inherits(x, "interceptspec")
}

#' @rdname is.varlse
#' @export
is.ssvsinput <- function(x) {
  inherits(x, "ssvsinput")
}

#' @rdname is.varlse
#' @export
is.ssvsinit <- function(x) {
  inherits(x, "ssvsinit")
}

#' @rdname is.varlse
#' @export
is.bvharpriorspec <- function(x) {
  inherits(x, "bvharpriorspec")
}

#' @rdname is.varlse
#' @export
is.horseshoespec <- function(x) {
  inherits(x, "horseshoespec")
}

#' @rdname is.varlse
#' @export
is.svspec <- function(x) {
  inherits(x, "svspec")
}
