.onLoad <- function(libname, pkgname) {
  Rcpp::registerPlugin(
    "bvhar",
    function() {
      list(env = list(PKG_CPPFLAGS = "DUSE_RCPP"))
    }
  )
}

.onAttach <- function(libname, pkgname) {
  if (!interactive()) {
    return()
  }
  packageStartupMessage(check_omp())
}

.onUnload <- function(libpath) {
  library.dynam.unload("bvhar", libpath)
}