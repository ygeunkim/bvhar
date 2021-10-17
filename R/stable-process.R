#' Roots of characteristic polynomial
#' 
#' @param x object
#' @param ... not used
#' 
#' @export
stableroot <- function(x, ...) {
  UseMethod("stableroot", x)
}

#' Stability of the process
#' 
#' @param x object
#' @param ... not used
#' 
#' @export
is.stable <- function(x, ...) {
  UseMethod("is.stable", x)
}

#' Characteristic polynomial roots for VAR Coefficient Matrix
#' 
#' Compute the character polynomial of VAR(p) coefficient matrix.
#' 
#' @param x Model fit
#' @param ... not used
#' @details 
#' To know whether the process is stable or not, make characteristic polynomial.
#' 
#' \deqn{\det(I_m - A z) = 0}
#' 
#' where \eqn{A} is VAR(1) matrix defined in [compute_stablemat()].
#' 
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @export
stableroot.varlse <- function(x, ...) {
  compute_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' Stability of VAR Coefficient Matrix
#' 
#' Check the stability condition of VAR(p) coefficient matrix.
#' 
#' @param x Model fit
#' @param ... not used
#' @details 
#' VAR(p) is stable if
#' 
#' \deqn{\det(I_m - A z) \neq 0}
#' 
#' for \eqn{\lvert z \rvert \le 1}.
#' 
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @export
is.stable.varlse <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
stableroot.vharlse <- function(x, ...) {
  compute_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
is.stable.vharlse <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
stableroot.bvarmn <- function(x, ...) {
  compute_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
is.stable.bvarmn <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
stableroot.bvarflat <- function(x, ...) {
  compute_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
is.stable.bvarflat <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
stableroot.bvharmn <- function(x, ...) {
  compute_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable.varlse
#' 
#' @param x Model fit
#' @param ... not used
#' 
#' @export
is.stable.bvharmn <- function(x, ...) {
  all(stableroot(x) < 1)
}