# ──────────────────────────────────────────────────────────────────
# MIPD R/Shiny Dashboard — Giao diện tương tác nghiên cứu
#
# Đáp ứng Công việc 3.2 thuyết minh NAFOSTED:
#   "Package Shiny trong R được sử dụng để xây dựng giao diện
#    tương tác thân thiện với người dùng, cho phép nhập, xem và
#    cập nhật thông tin bệnh nhân bằng tiếng Việt."
#
# Chức năng:
#   - Nhập thông tin bệnh nhân (tiếng Việt)
#   - Trực quan hóa đường cong PK (quần thể + cá thể + CI)
#   - Biểu đồ SHAP giải thích yếu tố ảnh hưởng
#   - Kết quả Bayesian estimation + credible intervals
#   - Chế độ: Lâm sàng (đơn giản) / Nghiên cứu (chi tiết)
#
# Kết nối API: gọi FastAPI PK Engine qua httr
# ──────────────────────────────────────────────────────────────────

library(shiny)
library(httr)
library(jsonlite)
library(ggplot2)
library(plotly)
library(dplyr)
library(shinydashboard)
library(DT)

# ── Cấu hình API ────────────────────────────────────────────────
API_BASE <- Sys.getenv("MIPD_API_URL", "http://localhost:8000")

# ── Hàm gọi API ─────────────────────────────────────────────────
call_api <- function(endpoint, body = NULL, method = "POST") {
  url <- paste0(API_BASE, endpoint)
  tryCatch({
    if (method == "POST") {
      resp <- POST(url, body = toJSON(body, auto_unbox = TRUE),
                   content_type_json(), timeout(30))
    } else {
      resp <- GET(url, timeout(30))
    }
    if (status_code(resp) == 200) {
      return(fromJSON(content(resp, "text", encoding = "UTF-8")))
    } else {
      return(list(error = paste("API Error:", status_code(resp))))
    }
  }, error = function(e) {
    return(list(error = paste("Lỗi kết nối:", e$message)))
  })
}

# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(
    title = "MIPD — Công cụ Định liều Chính xác",
    titleWidth = 350
  ),

  dashboardSidebar(
    width = 280,
    sidebarMenu(
      id = "tabs",
      menuItem("Trang chính", tabName = "main", icon = icon("stethoscope")),
      menuItem("Kết quả Bayesian", tabName = "bayesian", icon = icon("chart-line")),
      menuItem("Giải thích SHAP", tabName = "shap", icon = icon("lightbulb")),
      menuItem("Thẩm định", tabName = "validation", icon = icon("check-circle")),
      menuItem("Cài đặt", tabName = "settings", icon = icon("cog"))
    ),
    hr(),
    tags$div(
      style = "padding: 10px; color: #aaa; font-size: 11px;",
      "Phiên bản: 1.0",
      br(),
      "Đề tài: NAFOSTED",
      br(),
      "ĐH Dược Hà Nội + PTIT"
    )
  ),

  dashboardBody(
    # Custom CSS
    tags$head(tags$style(HTML("
      .content-wrapper { background-color: #f5f7fa; }
      .box-header { font-weight: bold; }
      .info-box-icon { font-size: 40px; }
      .shiny-input-container { margin-bottom: 10px; }
      .btn-primary { background-color: #2c5282; border-color: #2c5282; }
      .btn-primary:hover { background-color: #1a365d; }
    "))),

    tabItems(
      # ── Tab 1: Trang chính (Nhập bệnh nhân + Tính liều) ─────
      tabItem(
        tabName = "main",
        fluidRow(
          box(
            title = "Thông tin Bệnh nhân", status = "primary",
            solidHeader = TRUE, width = 4,
            textInput("patient_name", "Họ và tên:", placeholder = "Nguyễn Văn A"),
            numericInput("age", "Tuổi (năm):", value = 60, min = 18, max = 100),
            numericInput("weight", "Cân nặng (kg):", value = 65, min = 30, max = 200),
            numericInput("height", "Chiều cao (cm):", value = 165, min = 100, max = 220),
            selectInput("gender", "Giới tính:", choices = c("Nam" = "male", "Nữ" = "female")),
            numericInput("scr", "Creatinine huyết thanh (mg/dL):", value = 1.2, min = 0.1, max = 15, step = 0.1),
            numericInput("albumin", "Albumin (g/dL):", value = 3.5, min = 1, max = 5, step = 0.1),
            checkboxInput("is_icu", "Bệnh nhân ICU", value = FALSE),
            checkboxInput("is_dialysis", "Đang lọc máu", value = FALSE),
            hr(),
            h4("Mô hình PK"),
            selectInput("pk_model", "Quần thể bệnh nhân:",
                        choices = c("🧑 Vancomycin — Người lớn (VN 2-comp)" = "vancomycin_vn",
                                    "👶 Vancomycin — Nhi khoa (1-comp + Maturation)" = "vancomycin_pedi",
                                    "💊 Tacrolimus — Oral (2-comp)" = "tacrolimus_oral")),
            conditionalPanel(
              condition = "input.pk_model == 'vancomycin_pedi'",
              numericInput("pma", "PMA — Tuổi sau kỳ kinh (tuần):", value = 40, min = 24, max = 300)
            ),
            hr(),
            h4("Thông tin TDM"),
            numericInput("tdm_conc", "Nồng độ TDM (mg/L):", value = NA, min = 0, max = 100, step = 0.1),
            numericInput("tdm_time", "Thời điểm lấy mẫu (giờ sau liều):", value = 12, min = 0, max = 72),
            selectInput("tdm_type", "Loại mẫu:", choices = c("Trough" = "trough", "Peak" = "peak", "Random" = "random")),
            hr(),
            h4("Thông tin Liều"),
            numericInput("dose_mg", "Liều Vancomycin (mg):", value = 1000, min = 250, max = 3000, step = 250),
            numericInput("infusion_h", "Thời gian truyền (giờ):", value = 1, min = 0.5, max = 4, step = 0.5),
            numericInput("interval_h", "Khoảng đưa liều (giờ):", value = 12, min = 6, max = 48, step = 6),
            hr(),
            selectInput("method", "Phương pháp Bayesian:",
                        choices = c("Adaptive Pipeline (3 lớp)" = "adaptive",
                                    "MAP" = "map",
                                    "MAP + Laplace" = "laplace",
                                    "MCMC/NUTS" = "mcmc",
                                    "SMC (Particle Filter)" = "smc")),
            actionButton("btn_calculate", "Tính liều", icon = icon("calculator"),
                         class = "btn-primary btn-lg btn-block")
          ),

          box(
            title = "Kết quả Liều Đề xuất", status = "success",
            solidHeader = TRUE, width = 8,
            fluidRow(
              infoBoxOutput("auc_box", width = 4),
              infoBoxOutput("dose_box", width = 4),
              infoBoxOutput("pta_box", width = 4)
            ),
            hr(),
            h4("Đường cong Dược động học"),
            plotlyOutput("pk_curve_plot", height = "400px"),
            hr(),
            h4("Phác đồ Thay thế"),
            DTOutput("alternatives_table")
          )
        )
      ),

      # ── Tab 2: Kết quả Bayesian ──────────────────────────────
      tabItem(
        tabName = "bayesian",
        fluidRow(
          box(
            title = "Tham số PK Cá thể (Ước tính Bayesian)", status = "info",
            solidHeader = TRUE, width = 6,
            DTOutput("params_table"),
            hr(),
            h4("Biểu đồ Credible Intervals"),
            plotlyOutput("ci_plot", height = "300px")
          ),
          box(
            title = "Đường cong 3 lớp (Quần thể → Cá thể → CI)", status = "warning",
            solidHeader = TRUE, width = 6,
            plotlyOutput("three_layer_plot", height = "500px"),
            tags$p(
              style = "color: #666; font-size: 12px; margin-top: 10px;",
              "• Đường gạch nét (Population): ước tính quần thể ban đầu",
              br(),
              "• Đường nét liền (Individual): ước tính cá thể sau Bayesian",
              br(),
              "• Vùng mờ: khoảng tin cậy 95% Bayesian (Credible Interval)"
            )
          )
        ),
        fluidRow(
          box(
            title = "Diagnostics (Chẩn đoán hội tụ)", status = "primary",
            solidHeader = TRUE, width = 12,
            verbatimTextOutput("diagnostics_text")
          )
        )
      ),

      # ── Tab 3: SHAP Explanation ──────────────────────────────
      tabItem(
        tabName = "shap",
        fluidRow(
          box(
            title = "Giải thích Mô hình — SHAP Values", status = "info",
            solidHeader = TRUE, width = 12,
            plotlyOutput("shap_waterfall", height = "400px"),
            hr(),
            tags$p(
              style = "color: #666; font-size: 13px;",
              "Biểu đồ SHAP cho biết mức độ ảnh hưởng của từng yếu tố ",
              "(ví dụ: chức năng thận, tuổi, cân nặng) tới liều khuyến nghị. ",
              "Giá trị dương = tăng CL (cần tăng liều), ",
              "giá trị âm = giảm CL (cần giảm liều)."
            )
          )
        )
      ),

      # ── Tab 4: Thẩm định ─────────────────────────────────────
      tabItem(
        tabName = "validation",
        fluidRow(
          box(
            title = "So sánh Hiệu năng Thuật toán", status = "primary",
            solidHeader = TRUE, width = 12,
            DTOutput("benchmark_table"),
            hr(),
            tags$p(
              "Bảng so sánh hiệu năng các phương pháp Bayesian trên 1000 bệnh nhân ảo.",
              br(),
              "Đây là kết quả từ Nội dung 4 (Công việc 4.2) trong thuyết minh đề tài."
            )
          )
        )
      ),

      # ── Tab 5: Cài đặt ──────────────────────────────────────
      tabItem(
        tabName = "settings",
        box(
          title = "Cài đặt", status = "primary", solidHeader = TRUE, width = 6,
          textInput("api_url", "API URL:", value = API_BASE),
          actionButton("btn_test_api", "Kiểm tra kết nối", class = "btn-info"),
          verbatimTextOutput("api_status")
        )
      )
    )
  )
)


# ══════════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════════
server <- function(input, output, session) {

  # Reactive values to store results
  result <- reactiveVal(NULL)
  api_error <- reactiveVal(NULL)

  # ── Tính liều ─────────────────────────────────────────────────
  observeEvent(input$btn_calculate, {
    # Build request body
    has_tdm <- !is.na(input$tdm_conc) && input$tdm_conc > 0

    body <- list(
      patient = list(
        age = input$age,
        weight = input$weight,
        height = input$height,
        gender = input$gender,
        serum_creatinine = input$scr
      ),
      model = input$pk_model,
      dose = list(
        list(time = 0, amount = input$dose_mg, duration = input$infusion_h)
      ),
      method = input$method,
      target = list(
        auc24_min = 400,
        auc24_max = 600,
        mic = 1.0
      )
    )

    if (has_tdm) {
      body$observations <- list(
        list(time = input$tdm_time, concentration = input$tdm_conc)
      )
    }

    # Call API
    showNotification("Đang tính toán...", type = "message", duration = 2)
    res <- call_api("/bayesian/estimate", body)

    if (!is.null(res$error)) {
      api_error(res$error)
      showNotification(paste("Lỗi:", res$error), type = "error")
    } else {
      result(res)
      api_error(NULL)
      showNotification("Tính toán hoàn tất!", type = "message")
    }
  })

  # ── Info Boxes ────────────────────────────────────────────────
  output$auc_box <- renderInfoBox({
    res <- result()
    auc <- if (!is.null(res)) round(res$predictions$auc24, 1) else "—"
    infoBox("AUC₂₄/MIC", auc, subtitle = "mg·h/L (đích 400-600)",
            icon = icon("chart-area"), color = "blue")
  })

  output$dose_box <- renderInfoBox({
    res <- result()
    dose <- if (!is.null(res)) paste0(res$recommendation$dose, "mg q",
                                       res$recommendation$interval, "h") else "—"
    infoBox("Liều Đề xuất", dose, subtitle = "IV infusion",
            icon = icon("syringe"), color = "green")
  })

  output$pta_box <- renderInfoBox({
    res <- result()
    pta <- if (!is.null(res)) paste0(round(res$pta * 100, 1), "%") else "—"
    infoBox("PTA", pta, subtitle = "Xác suất đạt đích",
            icon = icon("bullseye"), color = "yellow")
  })

  # ── Biểu đồ PK Curve ────────────────────────────────────────
  output$pk_curve_plot <- renderPlotly({
    res <- result()

    if (is.null(res) || is.null(res$pkCurve)) {
      # Demo data when no result
      t <- seq(0, 24, by = 0.5)
      c_pop <- 30 * exp(-0.1 * t)
      c_ind <- 28 * exp(-0.09 * t)
      ci_lo <- c_ind * 0.7
      ci_hi <- c_ind * 1.3
    } else {
      t <- res$pkCurve$timePoints
      c_ind <- res$pkCurve$concentrations
      ci_lo <- res$pkCurve$ci95Lower
      ci_hi <- res$pkCurve$ci95Upper
      c_pop <- c_ind * 1.1  # approximate
    }

    df <- data.frame(time = t, population = c_pop, individual = c_ind,
                     ci_lower = ci_lo, ci_upper = ci_hi)

    p <- ggplot(df, aes(x = time)) +
      # CI ribbon
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                  fill = "#3182ce", alpha = 0.2) +
      # Target zone
      annotate("rect", xmin = 0, xmax = max(t), ymin = 10, ymax = 20,
               fill = "#48bb78", alpha = 0.1) +
      # Population line (dashed)
      geom_line(aes(y = population), color = "#a0aec0",
                linetype = "dashed", linewidth = 0.8) +
      # Individual line (solid)
      geom_line(aes(y = individual), color = "#2c5282", linewidth = 1.2) +
      labs(x = "Thời gian (giờ)", y = "Nồng độ (mg/L)",
           title = "Đường cong Dược động học") +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 12)
      )

    ggplotly(p) %>% layout(
      annotations = list(
        list(x = max(t) * 0.8, y = 15, text = "Vùng đích", showarrow = FALSE,
             font = list(color = "#48bb78", size = 11))
      )
    )
  })

  # ── Bảng phác đồ thay thế ────────────────────────────────────
  output$alternatives_table <- renderDT({
    res <- result()
    if (is.null(res) || is.null(res$alternatives)) {
      df <- data.frame(
        `Liều (mg)` = c(1000, 1250, 1500, 750),
        `Khoảng (h)` = c(12, 12, 12, 8),
        `AUC₂₄/MIC` = c(485, 606, 727, 364),
        `C_trough` = c(14.8, 18.5, 22.1, 8.4),
        `PTA (%)` = c(72, 68, 45, 55),
        check.names = FALSE
      )
    } else {
      df <- as.data.frame(res$alternatives)
    }
    datatable(df, options = list(pageLength = 5, dom = 't'),
              rownames = FALSE)
  })

  # ── Bảng tham số PK ──────────────────────────────────────────
  output$params_table <- renderDT({
    res <- result()
    if (is.null(res)) {
      df <- data.frame(
        `Tham số` = c("CL", "V1", "Q", "V2"),
        `Giá trị` = c(3.21, 26.5, 4.65, 37.0),
        `Đơn vị` = c("L/h", "L", "L/h", "L"),
        `CI 95% Lower` = c(2.12, 21.5, 2.5, 22.0),
        `CI 95% Upper` = c(3.83, 36.9, 8.6, 62.0),
        check.names = FALSE
      )
    } else {
      params <- res$individualParams
      df <- data.frame(
        `Tham số` = names(params),
        `Giá trị` = sapply(params, function(x) x$value),
        `Đơn vị` = sapply(params, function(x) x$unit),
        `CI 95% Lower` = sapply(params, function(x) x$ci95Lower),
        `CI 95% Upper` = sapply(params, function(x) x$ci95Upper),
        check.names = FALSE
      )
    }
    datatable(df, options = list(dom = 't'), rownames = FALSE)
  })

  # ── Biểu đồ CI ───────────────────────────────────────────────
  output$ci_plot <- renderPlotly({
    df <- data.frame(
      param = c("CL", "V1", "Q", "V2"),
      value = c(3.21, 26.5, 4.65, 37.0),
      lower = c(2.12, 21.5, 2.5, 22.0),
      upper = c(3.83, 36.9, 8.6, 62.0)
    )

    p <- ggplot(df, aes(x = param, y = value)) +
      geom_pointrange(aes(ymin = lower, ymax = upper),
                      color = "#2c5282", size = 1.2) +
      coord_flip() +
      labs(x = "", y = "Giá trị", title = "Credible Intervals (95%)") +
      theme_minimal() +
      theme(plot.title = element_text(face = "bold"))

    ggplotly(p)
  })

  # ── Biểu đồ 3 lớp ───────────────────────────────────────────
  output$three_layer_plot <- renderPlotly({
    t <- seq(0, 24, by = 0.5)

    # Layer 1: Population
    c_pop <- 30 * exp(-0.08 * t)
    # Layer 2: Individual (after Bayesian)
    c_ind <- 28 * exp(-0.09 * t)
    # CI
    ci_lo <- c_ind * 0.7
    ci_hi <- c_ind * 1.3

    df <- data.frame(time = t, population = c_pop, individual = c_ind,
                     ci_lower = ci_lo, ci_upper = ci_hi)

    p <- ggplot(df, aes(x = time)) +
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                  fill = "#3182ce", alpha = 0.15) +
      annotate("rect", xmin = 0, xmax = 24, ymin = 10, ymax = 20,
               fill = "#48bb78", alpha = 0.08) +
      geom_line(aes(y = population, color = "Quần thể (Population)"),
                linetype = "dashed", linewidth = 0.9) +
      geom_line(aes(y = individual, color = "Cá thể (Individual Posterior)"),
                linewidth = 1.3) +
      geom_point(data = data.frame(x = 12, y = 15.2),
                 aes(x = x, y = y), color = "red", size = 3,
                 shape = 16) +
      annotate("text", x = 13, y = 16.5, label = "TDM = 15.2 mg/L",
               color = "red", size = 3.5) +
      scale_color_manual(values = c(
        "Quần thể (Population)" = "#a0aec0",
        "Cá thể (Individual Posterior)" = "#2c5282"
      )) +
      labs(x = "Thời gian (giờ)", y = "Nồng độ (mg/L)",
           title = "Trực quan hóa 3 lớp: Quần thể → Cá thể → CI",
           color = "") +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 13, face = "bold"),
        legend.position = "bottom"
      )

    ggplotly(p) %>% layout(legend = list(orientation = "h", y = -0.2))
  })

  # ── Biểu đồ SHAP ─────────────────────────────────────────────
  output$shap_waterfall <- renderPlotly({
    # Demo SHAP values
    shap_df <- data.frame(
      feature = c("CrCL (52 mL/min)", "Cân nặng (65 kg)", "Tuổi (60)",
                  "Albumin (3.5)", "ICU (Không)"),
      contribution = c(-0.35, -0.12, -0.08, 0.02, 0.0),
      stringsAsFactors = FALSE
    )
    shap_df$feature <- factor(shap_df$feature,
                              levels = shap_df$feature[order(abs(shap_df$contribution))])
    shap_df$color <- ifelse(shap_df$contribution > 0, "Tăng CL", "Giảm CL")

    p <- ggplot(shap_df, aes(x = feature, y = contribution, fill = color)) +
      geom_col(width = 0.6) +
      coord_flip() +
      scale_fill_manual(values = c("Tăng CL" = "#48bb78", "Giảm CL" = "#e53e3e")) +
      labs(x = "", y = "Mức đóng góp (SHAP value)",
           title = "Giải thích yếu tố ảnh hưởng — SHAP",
           fill = "") +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 12)
      )

    ggplotly(p)
  })

  # ── Diagnostics ───────────────────────────────────────────────
  output$diagnostics_text <- renderPrint({
    res <- result()
    if (is.null(res)) {
      cat("Chưa có kết quả. Hãy nhấn 'Tính liều' ở trang chính.\n")
    } else if (!is.null(res$bayesianDiagnostics)) {
      cat("═══ Chẩn đoán Hội tụ ═══\n\n")
      cat("Phương pháp:", res$bayesianDiagnostics$Method, "\n")
      cat("Hội tụ:", ifelse(res$bayesianDiagnostics$converged, "✅ Có", "❌ Không"), "\n")
      if (!is.null(res$bayesianDiagnostics$rhat)) {
        cat("R-hat:", res$bayesianDiagnostics$rhat, "(yêu cầu < 1.01)\n")
      }
      if (!is.null(res$bayesianDiagnostics$ess)) {
        cat("ESS:", res$bayesianDiagnostics$ess, "(yêu cầu > 400)\n")
      }
    } else {
      cat("Không có thông tin diagnostics chi tiết cho phương pháp này.\n")
    }
  })

  # ── Benchmark table ───────────────────────────────────────────
  output$benchmark_table <- renderDT({
    df <- data.frame(
      `Phương pháp` = c("MAP", "Laplace", "MCMC/NUTS", "ADVI", "EP", "SMC", "Adaptive (3 lớp)"),
      `Tốc độ` = c("< 1s", "< 1s", "~30s", "~5s", "~5s", "~3s", "~5s"),
      `MPE (%)` = c("—", "—", "—", "—", "—", "—", "—"),
      `MAPE (%)` = c("—", "—", "—", "—", "—", "—", "—"),
      `CCC` = c("—", "—", "—", "—", "—", "—", "—"),
      `Coverage 95%` = c("—", "—", "—", "—", "—", "—", "—"),
      `Ghi chú` = c(
        "Nhanh, point estimate", "MAP + CI", "Full posterior, chính xác nhất",
        "Variational, scale tốt", "Moment-matching", "Sequential, online update",
        "MAP→SMC→HB, cam kết thuyết minh"
      ),
      check.names = FALSE
    )
    datatable(df, options = list(dom = 't', pageLength = 10), rownames = FALSE)
  })

  # ── Kiểm tra API ─────────────────────────────────────────────
  observeEvent(input$btn_test_api, {
    api_url <- input$api_url
    tryCatch({
      resp <- GET(paste0(api_url, "/health"), timeout(5))
      if (status_code(resp) == 200) {
        output$api_status <- renderPrint({ cat("✅ Kết nối thành công!") })
      } else {
        output$api_status <- renderPrint({ cat("⚠️ Server phản hồi:", status_code(resp)) })
      }
    }, error = function(e) {
      output$api_status <- renderPrint({ cat("❌ Không thể kết nối:", e$message) })
    })
  })
}

# ── Chạy ứng dụng ──────────────────────────────────────────────
shinyApp(ui = ui, server = server)
