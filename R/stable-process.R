#' Roots of characteristic polynomial
#' 
#' Compute the character polynomial of coefficient matrix.
#' 
#' @param x Model fit
#' @param ... not used
#' @return Numeric vector.
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @export
stableroot <- function(x, ...) {
  UseMethod("stableroot", x)
}

#' Stability of the process
#' 
#' Check the stability condition of coefficient matrix.
#' 
#' @param x Model fit
#' @param ... not used
#' @return logical class
#' @export
is.stable <- function(x, ...) {
  UseMethod("is.stable", x)
}

#' @rdname stableroot
#' @details 
#' To know whether the process is stable or not, make characteristic polynomial.
#' 
#' \deqn{\det(I_m - A z) = 0}
#' 
#' where \eqn{A} is VAR(1) coefficient matrix representation.
#' @export
stableroot.varlse <- function(x, ...) {
  compute_var_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable
#' @details 
#' VAR(p) is stable if
#' 
#' \deqn{\det(I_m - A z) \neq 0}
#' 
#' for \eqn{\lvert z \rvert \le 1}.
#' @return logical class
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @export
is.stable.varlse <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot
#' @export
stableroot.vharlse <- function(x, ...) {
  compute_vhar_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable
#' @export
is.stable.vharlse <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot
#' @export
stableroot.bvarmn <- function(x, ...) {
  compute_var_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable
#' @export
is.stable.bvarmn <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot
#' @export
stableroot.bvarflat <- function(x, ...) {
  compute_var_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable
#' @export
is.stable.bvarflat <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' @rdname stableroot
#' @export
stableroot.bvharmn <- function(x, ...) {
  compute_vhar_stablemat(x) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' @rdname is.stable
#' @export
is.stable.bvharmn <- function(x, ...) {
  all(stableroot(x) < 1)
}
