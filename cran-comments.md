## Minor version update

In this version, we

- Added `exogen` features.
- Included some internal C++ changes.
- Require `R >= 4.2` now.

## Test environments

- Local: macOS 15.5 (aarch64-apple-darwin20), R 4.5.1
- Github actions
    - ubuntu-latest: R-devel, R-release, R-oldrel-1, R-oldrel-2, R-oldrel-3
    - macOS-latest: R-release
    - windows-latest: R-release, R-oldrel-3
- win-builder: devel

## R CMD check results

0 errors | 0 warnings | 0 notes

## revdepcheck results

We checked 0 reverse dependencies, comparing R CMD check results across CRAN and dev versions of this package.

 * We saw 0 new problems
 * We failed to check 0 packages