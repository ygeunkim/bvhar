## Minor version update

In this version, we

- Fix some numerical process
- Fix a C++ macro issue

## Test environments

- Local: macOS 14.6.1 (x86_64-apple-darwin20), R 4.3.3
- Github actions
    - ubuntu-latest: R-devel, R-release, R-oldrel-1, R-oldrel-2, R-oldrel-3, R 3.6
    - macOS-latest: R-release
    - windows-latest: R-release, R-oldrel-4
- win-builder: devel

## R CMD check results

0 errors | 0 warnings | 1 note

* NOTE in local machine: HTML validation NOTE on local environment (aarch64-apple-darwin20) check. This note appears to be specific to my local machine and had no problem in previous CRAN checks.

## revdepcheck results

We checked 0 reverse dependencies, comparing R CMD check results across CRAN and dev versions of this package.

 * We saw 0 new problems
 * We failed to check 0 packages