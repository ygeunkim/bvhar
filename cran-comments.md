## Patch version update

This is a quick fix for fatal algorithm fault.
In this version, we

- Fixed MCMC algorithm when constant term exists.
- Fixed Forecasting algorithm for MCMC objects.

## Test environments

- Local: macOS 14.6.1 (x86_64-apple-darwin20), R 4.3.3
- Github actions
    - ubuntu-latest: R-devel, R-release, R-oldrel-1, R-oldrel-2, R-oldrel-3, R 3.6
    - macOS-latest: R-release
    - windows-latest: R-release, R-oldrel-4
- win-builder: devel

## R CMD check results

0 errors | 0 warnings | 1 note

* NOTE for CRAN incoming feasibility (Days since last update: 6):
The algorithms in the current version have some serious issues,
so an update is needed ASAP.
* NOTE only in local machine: HTML validation NOTE on local environment check. This note appears to be specific to my local machine and had no problem in previous CRAN checks.

## revdepcheck results

We checked 0 reverse dependencies, comparing R CMD check results across CRAN and dev versions of this package.

 * We saw 0 new problems
 * We failed to check 0 packages