## Re-resubmission v1.0.0
This is second resubmission. Comments are included in italics.

- *We still see: Missing Rd-tags ... Since you are using 'roxygen', please add a @return tag in the corresponding .R-file and re-roxygenize() your .Rd-files.*\
**Response** We have added `@return` to each mentioned document.

## Resubmission
This is a resubmission. Comments are included in italics.

- *Please proof-read your description text. Currently it reads: " Aim at researching Bayesian Vector heterogeneous autoregressive (VHAR) model. ..." Probably it should be: " Aims at researching Bayesian Vector heterogeneous autoregressive (VHAR) models. ..." Furthermore, please write a full sentence for the second sentence in your description text.*\
**Response** Done.
- *If there are references describing the methods in your package, please add these in the description field of your DESCRIPTION file in the form authors (year) <doi:...>, authors (year) <arXiv:...>, authors (year, ISBN:...), or if those are not available: <https:...> with no space after 'doi:', 'arXiv:', 'https:' and angle brackets for auto-linking. (If you want to add a title as well please put it in quotes: "Title")*\
**Response** We have added the references and its doi, but doi has not been activated. DOI in DESCRIPTION has been given by publisher. It will be permanently available after online publish, soon. That's why we have written its year by 2023+.
- *Please add \value to .Rd files regarding exported methods and explain the functions results in the documentation. Please write about the structure of the output (class) and also what the output means. (If a function does not return a value, please document that too, e.g. \value{No return value, called for side effects} or similar)*\
**Response** Done. Now every document has \value{} section.
- *Please always make sure to reset to user's options(), working directory or par() after you changed it in examples and vignettes and demos.*\
**Response** We have reset `options(digits = 3)` in vignettes by suggested methods.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## revdepcheck results

We checked 0 reverse dependencies, comparing R CMD check results across CRAN and dev versions of this package.

 * We saw 0 new problems
 * We failed to check 0 packages