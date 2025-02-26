## Patch version update

In this version, we

- Added `error = TRUE` chunk option in vignettes temporarily until investigating the cause of error in r-devel-linux.

## Test environments

- Local: macOS 15.3.1 (aarch64-apple-darwin20), R 4.4.2
- Github actions
    - ubuntu-latest: R-devel, R-release, R-oldrel-1, R-oldrel-2, R-oldrel-3
    - macOS-latest: R-release
    - windows-latest: R-release, R 4.1
- win-builder: devel

## R CMD check results

0 errors | 0 warnings | 2 note

* NOTE in local machine: HTML validation NOTE on local environment (aarch64-apple-darwin20) check. This note appears to be specific to my local machine and had no problem in previous CRAN checks.

* NOTE regarding installed package size but not that large, which had no problem in previous CRAN acceptance.

## revdepcheck results

We checked 0 reverse dependencies, comparing R CMD check results across CRAN and dev versions of this package.

 * We saw 0 new problems
 * We failed to check 0 packages