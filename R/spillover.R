#' h-step ahead Normalized Spillover
#'
#' This function gives connectedness table with h-step ahead normalized spillover index (a.k.a. variance shares).
#'
#' @param object Model object
#' @param n_ahead step to forecast
#' @param num_iter Number to sample MNIW distribution
#' @param num_burn Number of burn-in
#' @param thinning Thinning every thinning-th iteration
#' @param ... Not used
#' @importFrom tibble rownames_to_column
#' @importFrom tidyr pivot_longer
#' @order 1
#' @export
spillover_volatility <- function(object,
                                 n_ahead,
                                 num_iter = 10000L,
                                 num_burn = floor(num_iter / 2),
                                 thinning = 1L, ...) {
  UseMethod("spillover_volatility", object)
}

#' @rdname spillover_volatility
#' @export 
spillover_volatility.bvharmod <- function(object,
                                          n_ahead,
                                          num_iter = 10000L,
                                          num_burn = floor(num_iter / 2),
                                          thinning = 1L, ...) {
  if (object$process == "VAR") {
    mod_type <- "freq_var"
  } else if (object$process == "VHAR") {
    mod_type <- "freq_vhar"
  } else {
    mod_type <- ifelse(grepl(pattern = "^BVAR_", object$process), "var", "vhar")
  }
  dim_data <- object$coefficients
  if (grepl(pattern = "^freq_", mod_type)) {
    vma_coef <- switch(mod_type,
      "freq_var" = VARtoVMA(object, n_ahead - 1),
      "freq_vhar" = VHARtoVMA(object, n_ahead - 1)
    )
    fevd <- compute_fevd(vma_coef, object$covmat)
    connect_tab <- compute_spillover(fevd)
    colnames(connect_tab) <- colnames(object$coefficients)
    rownames(connect_tab) <- colnames(object$coefficients)
    connect_tab
    sp_long <-
      connect_tab %>%
      as.data.frame() %>%
      rownames_to_column(var = "shock") %>%
      pivot_longer(-"shock", names_to = "series", values_to = "spillover")
    res <- list(
      connect = connect_tab,
      df_long = sp_long
    )
  } else {
    mn_mean <- object$coefficients
    coef_and_sig <- sim_mniw(
      num_iter,
      mn_mean,
      solve(object$mn_prec),
      object$iw_scale,
      object$iw_shape
    )
    thin_id <- seq(from = num_burn + 1, to = num_iter, by = thinning)
    coef_record <-
      coef_and_sig$mn %>%
      t() %>%
      split.data.frame(gl(num_iter, object$m)) %>%
      lapply(function(x) t(x))
    coef_record <- coef_record[thin_id]
    cov_record <- split_psirecord(t(coef_and_sig$iw), chain = 1, varname = "psi")
    cov_record <- cov_record[thin_id]
    mod_type <- ifelse(grepl(pattern = "^BVAR_", object$process), "var", "vhar")
    sp_list <- lapply(
      seq_along(thin_id),
      function(x) {
        vma_coef <- switch(mod_type,
          "var" = VARcoeftoVMA(coef_record[[x]], object$p, n_ahead - 1),
          "vhar" = VHARcoeftoVMA(coef_record[[x]], object$HARtrans, n_ahead - 1, object$month)
        )
        fevd <- compute_fevd(coef_record[[x]], cov_record[[x]])
        connect_tab <- compute_spillover(fevd)
        colnames(connect_tab) <- colnames(mn_mean)
        rownames(connect_tab) <- colnames(mn_mean)
        connect_tab
      }
    )
    sp_long <- lapply(
      seq_along(sp_list),
      function(id) {
        sp_list[[id]] %>%
          as.data.frame() %>%
          rownames_to_column(var = "shock") %>%
          pivot_longer(-"shock", names_to = "series", values_to = "spillover") %>%
          mutate(draws_id = id)
      }
    )
    sp_long <- do.call(rbind, sp_long)
    sp_mean <- Reduce("+", sp_list) / length(sp_list)
    res <- list(
      connect = sp_mean,
      df_long = sp_long,
      record = sp_list
    )
  }
  res$process <- object$process
  class(res) <- "bvharspillover"
  res
}

#' Summarizing Spillover
#' 
#' This function computes directional spillovers.
#' 
#' @param object `bvharspillover` object
#' @param ... Not used
#' @order 1
#' @export
summary.bvharspillover <- function(object, ...) {
  if (object$process == "VAR") {
    mod_type <- "freq_var"
  } else if (object$process == "VHAR") {
    mod_type <- "freq_vhar"
  } else {
    mod_type <- ifelse(grepl(pattern = "^BVAR_", object$process), "var", "vhar")
  }
  sp_tab <- object$connect
  if (grepl(pattern = "^freq_", mod_type)) {
    to_others <- compute_to_spillover(sp_tab)
    from_others <- compute_from_spillover(sp_tab)
    tot_sp <- compute_tot_spillover(sp_tab)
  } else {
    to_list <- lapply(
      object$record,
      compute_to_spillover
    ) %>%
      do.call(rbind, .)
    from_list <- lapply(
      object$record,
      compute_from_spillover
    ) %>%
      do.call(rbind, .)
    tot_list <- sapply(object$record, compute_tot_spillover)
    to_others <- colMeans(to_list)
    from_others <- colMeans(from_list)
    tot_sp <- mean(tot_list)
  }
  sp_tab <- rbind(sp_tab, "from others" = from_others)
  sp_tab <- cbind(sp_tab, "to others" = c(to_others, tot_sp))
  net_sp <- to_others - from_others
  res <- list(
    connect = sp_tab,
    df_long = object$sp_long,
    to = to_others,
    from = from_others,
    tot = tot_sp,
    net = net_sp
  )
  if (!grepl(pattern = "^freq_", mod_type)) {
    res$to_record <- to_list
    res$from_record <- from_list
    res$tot_record <- tot_list
  }
  class(res) <- "summary.bvharspillover"
  res
}
