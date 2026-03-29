const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;

const parseCsv = (text) => {
  const rows = [];
  let current = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      value += '"';
      i += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === ',' && !inQuotes) {
      current.push(value);
      value = "";
      continue;
    }

    if (char === '\n' && !inQuotes) {
      current.push(value);
      rows.push(current);
      current = [];
      value = "";
      continue;
    }

    value += char;
  }

  if (value.length || current.length) {
    current.push(value);
    rows.push(current);
  }

  return rows;
};

const buildImageCard = (src, caption) => {
  const figure = document.createElement("figure");
  figure.className = "figure-card";
  const img = document.createElement("img");
  img.src = src;
  img.alt = caption;
  const figcaption = document.createElement("figcaption");
  figcaption.textContent = caption;
  figure.append(img, figcaption);
  return figure;
};

const renderTable = (rows, headers) => {
  const tableHead = document.getElementById("tableHead");
  const tableBody = document.getElementById("tableBody");

  tableHead.innerHTML = "";
  tableBody.innerHTML = "";

  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  tableHead.appendChild(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    headers.forEach((header) => {
      const td = document.createElement("td");
      td.textContent = row[header] ?? "";
      tr.appendChild(td);
    });
    tableBody.appendChild(tr);
  });
};

const filterRows = (rows, query) => {
  if (!query) return rows;
  const needle = query.toLowerCase();
  return rows.filter((row) =>
    Object.values(row).some((value) => String(value).toLowerCase().includes(needle))
  );
};

const main = async () => {
  const summary = await fetch("data/summary.json").then((response) => response.json());
  const reportText = await fetch("data/00_Study_Report.txt").then((response) => response.text());
  document.getElementById("reportText").textContent = reportText;

  const availableTarget = summary.targets.find((target) => target.available) || summary.targets[0];
  const updatedAt = new Date(summary.generated_at);
  document.getElementById("updatedAt").textContent = `Updated ${updatedAt.toLocaleString()}`;
  document.getElementById("targetBadge").textContent = availableTarget.target_name || "Target";
  document.getElementById("ed1Badge").textContent = summary.cohort.ed1_available
    ? "ED I available"
    : "ED I unavailable";

  document.getElementById("totalPatients").textContent = summary.cohort.total_patients;
  document.getElementById("prevalence").textContent = formatPercent(availableTarget.prevalence || 0);
  document.getElementById("rocAuc").textContent = (availableTarget.roc_auc ?? 0).toFixed(3);
  document.getElementById("avgPrecision").textContent = (availableTarget.avg_precision ?? 0).toFixed(3);

  const kpiConfig = [
    { label: "Accuracy", value: formatPercent(availableTarget.accuracy || 0) },
    { label: "Sensitivity", value: formatPercent(availableTarget.sensitivity || 0) },
    { label: "Specificity", value: formatPercent(availableTarget.specificity || 0) },
    { label: "PPV", value: formatPercent(availableTarget.ppv || 0) },
    { label: "NPV", value: formatPercent(availableTarget.npv || 0) },
    { label: "Train / Test", value: `${availableTarget.train_n} / ${availableTarget.test_n}` },
  ];

  const kpiGrid = document.getElementById("kpiGrid");
  kpiConfig.forEach((item) => {
    const card = document.createElement("div");
    card.className = "kpi-card";
    const title = document.createElement("h3");
    title.textContent = item.label;
    const value = document.createElement("p");
    value.textContent = item.value;
    card.append(title, value);
    kpiGrid.appendChild(card);
  });

  const overviewImages = document.getElementById("overviewImages");
  const overviewCaptions = {
    "01_Cohort_Overview.png": "Cohort distributions (age, weight, surgery type, mYPAS)",
    "02_Target_Prevalence.png": "Target prevalence in the current cohort",
    "03_Feature_Correlations.png": "Feature correlation heatmap",
  };

  summary.assets_common.forEach((asset) => {
    const caption = overviewCaptions[asset] || asset;
    overviewImages.appendChild(buildImageCard(`assets/${asset}`, caption));
  });

  const diagnosticImages = document.getElementById("diagnosticImages");
  if (availableTarget.assets) {
    const targetCaptions = {
      [availableTarget.assets.confusion_matrix]: "Confusion matrix",
      [availableTarget.assets.roc_curve]: "ROC curve",
      [availableTarget.assets.precision_recall]: "Precision-recall curve",
      [availableTarget.assets.feature_importance]: "Top feature importance",
      [availableTarget.assets.performance_summary]: "Performance summary metrics",
      [availableTarget.assets.risk_distribution]: "Predicted risk distribution",
    };
    Object.entries(availableTarget.assets).forEach(([key, asset]) => {
      if (!asset || asset.endsWith(".csv") || key === "ai_vs_clinician") return;
      const caption = targetCaptions[asset] || asset;
      diagnosticImages.appendChild(buildImageCard(`assets/${asset}`, caption));
    });
  }

  const comparisonGrid = document.getElementById("comparisonGrid");
  const clinicianMetrics = summary.comparison?.clinician_full_metrics;
  if (comparisonGrid && clinicianMetrics) {
    const aiMetrics = {
      Accuracy: availableTarget.accuracy,
      Sensitivity: availableTarget.sensitivity,
      Specificity: availableTarget.specificity,
      PPV: availableTarget.ppv,
      NPV: availableTarget.npv,
    };
    const clinicianDisplay = {
      Accuracy: clinicianMetrics.accuracy,
      Sensitivity: clinicianMetrics.sensitivity,
      Specificity: clinicianMetrics.specificity,
      PPV: clinicianMetrics.ppv,
      NPV: clinicianMetrics.npv,
    };

    Object.keys(aiMetrics).forEach((metric) => {
      const card = document.createElement("div");
      card.className = "kpi-card";
      const title = document.createElement("h3");
      title.textContent = metric;
      const value = document.createElement("p");
      value.textContent = `${formatPercent(aiMetrics[metric] || 0)} vs ${formatPercent(
        clinicianDisplay[metric] || 0
      )}`;
      const caption = document.createElement("span");
      caption.className = "caption";
      caption.textContent = "AI vs Anesthesiologist";
      card.append(title, value, caption);
      comparisonGrid.appendChild(card);
    });
  }

  const comparisonImages = document.getElementById("comparisonImages");
  if (comparisonImages && availableTarget.assets?.ai_vs_clinician) {
    comparisonImages.appendChild(
      buildImageCard(
        `assets/${availableTarget.assets.ai_vs_clinician}`,
        "AI vs anesthesiologist performance"
      )
    );
  }

  if (availableTarget.assets?.scoreboard) {
    const csvText = await fetch(`data/${availableTarget.assets.scoreboard}`).then((response) =>
      response.text()
    );
    const rows = parseCsv(csvText);
    const headers = rows[0];
    const dataRows = rows
      .slice(1)
      .map((row) =>
        headers.reduce((acc, header, index) => ({ ...acc, [header]: row[index] }), {})
      );

    const sortedRows = [...dataRows].sort((a, b) => {
      const aValue = parseFloat(a.AI_Confidence || 0);
      const bValue = parseFloat(b.AI_Confidence || 0);
      return bValue - aValue;
    });

    let filteredRows = [...sortedRows];
    const displayHeaders = [
      "Age",
      "Gender",
      "Surgery Type",
      "Preop mYPAS score",
      "Duration of surgery (mins)",
      "Time to emergence (mins)",
      "Actual_Outcome",
      "AI_Prediction",
      "AI_Confidence",
      "Anesthesiologist_Prediction",
      "Anesthesiologist_Correct",
    ].filter((header) => headers.includes(header));

    renderTable(filteredRows.slice(0, 25), displayHeaders);

    const filterInput = document.getElementById("tableFilter");
    const resetButton = document.getElementById("resetFilter");

    const applyFilter = () => {
      const query = filterInput.value.trim();
      filteredRows = filterRows(sortedRows, query);
      renderTable(filteredRows.slice(0, 25), displayHeaders);
    };

    filterInput.addEventListener("input", applyFilter);
    resetButton.addEventListener("click", () => {
      filterInput.value = "";
      applyFilter();
    });
  }
};

main();
