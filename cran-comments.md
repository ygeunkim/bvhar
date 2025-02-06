## Patch version update

This is a quick fix for fatal algorithm fault.
In this version, we

- Fixed MCMC algorithm when constant term exists.
- Fixed Forecasting algorithm for MCMC objects.

## Test environments

- Local: macOS 15.3 (aarch64-apple-darwin20), R 4.4.2
- Github actions
    - ubuntu-latest: R-devel, R-release, R-oldrel-1, R-oldrel-2, R-oldrel-3
    - macOS-latest: R-release
    - windows-latest: R-release, R 4.1
- win-builder: devel

## R CMD check results

0 errors | 0 warnings | 1 note

* NOTE regarding installed package size but not that large, which had no problem in previous CRAN acceptance.

## revdepcheck results

We checked 0 reverse dependencies, comparing R CMD check results across CRAN and dev versions of this package.

 * We saw 0 new problems
 * We failed to check 0 packages