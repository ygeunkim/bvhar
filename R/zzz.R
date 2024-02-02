.onAttach <- function(libname, pkgname) {
  if (!interactive()) {
    return()
  }
  packageStartupMessage(check_omp())
}

.onUnload <- function(libpath) {
  library.dynam.unload("bvhar", libpath)
}