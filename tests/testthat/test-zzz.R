# Package message-------------------
test_that("Package attach message", {
  if (interactive()) {
    print_output <- capture.output(.onAttach(libname = "dummy_lib", pkgname = "bvhar"))
    if (is_omp()) {
      expect_true(grepl("OpenMP threads: [0-9]+", print_output))
    } else {
      expect_true(grepl("OpenMP not available in this machine", print_output))
    }
  } else {
    skip(".onAttach() prints nothing in non-interactive.")
  }
})
#> Test passed 🌈