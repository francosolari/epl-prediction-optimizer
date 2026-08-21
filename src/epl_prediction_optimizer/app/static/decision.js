(() => {
  document.documentElement.classList.add("js");
  const root = document.documentElement;
  const themeButton = document.querySelector("[data-theme-toggle]");
  const themes = ["system", "light", "dark"];
  const stored = localStorage.getItem("eplpo-theme") || "system";
  root.dataset.theme = stored;
  if (themeButton) themeButton.textContent = stored[0].toUpperCase() + stored.slice(1);
  themeButton?.addEventListener("click", () => {
    const next = themes[(themes.indexOf(root.dataset.theme) + 1) % themes.length];
    root.dataset.theme = next;
    localStorage.setItem("eplpo-theme", next);
    themeButton.textContent = next[0].toUpperCase() + next.slice(1);
  });

  document.querySelector("[data-week-picker]")?.addEventListener("change", (event) => {
    window.location.href = `/?week=${event.target.value}`;
  });

  const dataNode = document.querySelector("#scenario-data");
  const scenarios = dataNode ? JSON.parse(dataNode.textContent) : [];
  const rows = [...document.querySelectorAll("[data-scenario-id]")];
  const chart = document.querySelector("[data-forecast-chart]");
  const pathsGroup = chart?.querySelector(".chart-paths");
  const gridGroup = chart?.querySelector(".chart-grid");
  const markersGroup = chart?.querySelector(".chart-markers");
  const moveTree = document.querySelector("[data-move-tree]");
  const baselinePlan = moveTree ? [...moveTree.querySelectorAll("[data-week]")].map((node) => ({
    week: Number(node.dataset.week), team: node.dataset.team, ev: Number(node.dataset.ev)
  })) : [];
  const commitButton = document.querySelector("[data-commit-button]");
  let selected = scenarios.find((item) => item.selected) || scenarios.find((item) => item.season_rank === 1);

  function escapeHtml(value) {
    const node = document.createElement("span");
    node.textContent = String(value);
    return node.innerHTML;
  }

  function curvedPath(points) {
    if (!points.length) return "";
    if (points.length === 1) return `M${points[0].x},${points[0].y}`;
    return points.slice(1).reduce((path, point, index) => {
      const previous = points[index];
      const middle = (previous.x + point.x) / 2;
      return `${path} C${middle},${previous.y} ${middle},${point.y} ${point.x},${point.y}`;
    }, `M${points[0].x},${points[0].y}`);
  }

  function drawChart() {
    if (!pathsGroup || !gridGroup || !markersGroup) return;
    const feasible = scenarios.filter((scenario) => scenario.feasible && scenario.horizon?.length);
    const weeks = [...new Set(feasible.flatMap((scenario) => scenario.horizon.map((point) => point.week)))].sort((a, b) => a - b);
    const bestAtWeek = new Map(weeks.map((week) => [week, Math.max(...feasible.map((scenario) => scenario.horizon.find((point) => point.week === week)?.value ?? -Infinity))]));
    const relative = feasible.map((scenario) => ({
      scenario,
      points: scenario.horizon.map((point) => ({week: point.week, value: point.value - bestAtWeek.get(point.week)}))
    }));
    const depth = Math.max(0.25, ...relative.flatMap((line) => line.points.map((point) => Math.abs(point.value))));
    const xFor = (week) => 46 + (weeks.indexOf(week) / Math.max(weeks.length - 1, 1)) * 642;
    const yFor = (value) => 38 + (Math.abs(value) / depth) * 214;
    const coordinates = (points) => points.map((point) => ({x: xFor(point.week).toFixed(1), y: yFor(point.value).toFixed(1), ...point}));

    gridGroup.innerHTML = [
      `<line class="zero-line" x1="46" x2="688" y1="38" y2="38"/><text x="8" y="42">0.00</text>`,
      `<line x1="46" x2="688" y1="145" y2="145"/><text x="4" y="149">−${(depth / 2).toFixed(2)}</text>`,
      `<line x1="46" x2="688" y1="252" y2="252"/><text x="4" y="256">−${depth.toFixed(2)}</text>`,
      `<text class="axis-caption" x="46" y="286">W${weeks[0] ?? "—"}</text><text class="axis-caption axis-end" x="688" y="286">W${weeks.at(-1) ?? "—"}</text>`
    ].join("");
    document.querySelector("[data-chart-scale]").textContent = `Relative season edge · zoomed 0 to −${depth.toFixed(2)} xPts`;
    pathsGroup.innerHTML = relative.map(({scenario, points}) => {
      const active = selected && scenario.match_id === selected.match_id && scenario.team === selected.team;
      const kind = scenario.season_rank === 1 ? "season" : scenario.immediate_rank === 1 ? "now" : "alt";
      return `<path class="scenario-path ${kind} ${active ? "active" : ""}" d="${curvedPath(coordinates(points))}"><title>${escapeHtml(scenario.team)}: ${scenario.season_cost.toFixed(2)} expected-point cost across the active season line</title></path>`;
    }).join("");
    const selectedLine = relative.find(({scenario}) => selected && scenario.match_id === selected.match_id && scenario.team === selected.team);
    const affected = selectedLine?.points.find((point) => point.week === selected.first_affected_week);
    markersGroup.innerHTML = affected ? `<line class="impact-guide" x1="${xFor(affected.week)}" x2="${xFor(affected.week)}" y1="38" y2="252"/><circle cx="${xFor(affected.week)}" cy="${yFor(affected.value)}" r="5"/><text x="${Math.min(xFor(affected.week) + 9, 620)}" y="${Math.max(yFor(affected.value) - 10, 22)}">First reroute · W${affected.week}</text>` : "";
  }

  function renderMoveTree() {
    if (!moveTree || !selected || !baselinePlan.length) return;
    const changes = new Map((selected.changes || []).map((change) => [Number(change.week), change]));
    const currentWeek = Number(selected.contest_week);
    const upcoming = baselinePlan.filter((pick) => pick.week >= currentWeek).slice(0, 5).map((pick) => pick.week);
    const finalWeek = baselinePlan.at(-1)?.week;
    const keyWeeks = [...new Set([currentWeek, ...upcoming, ...changes.keys(), finalWeek])].filter(Boolean).sort((a, b) => a - b);
    const horizon = new Map();
    let previous = 0;
    (selected.horizon || []).forEach((point) => {
      horizon.set(Number(point.week), Number(point.value) - previous);
      previous = Number(point.value);
    });
    let lastWeek = null;
    moveTree.innerHTML = keyWeeks.map((week) => {
      const base = baselinePlan.find((pick) => pick.week === week);
      const change = changes.get(week);
      const team = week === currentWeek ? selected.team : change?.to || base?.team || "Open";
      const gap = lastWeek !== null && week - lastWeek > 1 ? `<span class="route-gap" aria-label="Weeks ${lastWeek + 1} through ${week - 1} continue unchanged">+${week - lastWeek - 1}</span>` : "";
      lastWeek = week;
      const branch = change && change.from !== team ? `<span class="route-branch"><small>Original line</small><s>${escapeHtml(change.from)}</s></span>` : "";
      const node = `<span class="route-step ${week === currentWeek ? "is-current" : ""} ${change ? "is-rerouted" : ""}"><a href="/?week=${week}"><small>W${week}</small><strong>${escapeHtml(team)}</strong><em>${(horizon.get(week) ?? base?.ev ?? 0).toFixed(2)} xPts</em></a>${branch}</span>`;
      return `${gap}${node}`;
    }).join("");
  }

  function renderSelection() {
    if (!selected) return;
    rows.forEach((row) => {
      const active = row.dataset.scenarioId === `${selected.match_id}-${selected.team.replaceAll(" ", "-")}`;
      row.classList.toggle("is-selected", active);
      row.setAttribute("aria-checked", active ? "true" : "false");
    });
    document.querySelector("[data-summary-title]").textContent = `${selected.team} vs ${selected.opponent}`;
    document.querySelector("[data-summary-label]").textContent = selected.season_rank === 1 ? "Best season line" : "Alternative line";
    const impact = selected.first_affected_week ? `First changes week ${selected.first_affected_week}.` : "No downstream plan change.";
    document.querySelector("[data-summary-copy]").textContent = `${selected.season_cost.toFixed(2)} season xPts cost. ${impact}`;
    document.querySelector("[data-commit-summary]").textContent = `${selected.team} · ${selected.expected_points.toFixed(2)} xPts now`;
    document.querySelector("[data-impact-copy]").textContent = impact;
    if (commitButton && !commitButton.disabled) commitButton.textContent = `Commit ${selected.team}`;
    drawChart();
    renderMoveTree();
  }

  rows.forEach((row) => {
    const activate = () => {
      const id = row.dataset.scenarioId;
      selected = scenarios.find((s) => `${s.match_id}-${s.team.replaceAll(" ", "-")}` === id);
      renderSelection();
    };
    row.addEventListener("click", (event) => { if (!event.target.closest("button")) activate(); });
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); } });
  });

  commitButton?.addEventListener("click", async () => {
    if (!selected || commitButton.disabled) return;
    commitButton.disabled = true;
    commitButton.textContent = "Committing…";
    try {
      const response = await fetch(`/api/picks/${new URLSearchParams(location.search).get("week") || document.querySelector("[data-week-picker]")?.value}`, {
        method: "PUT", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({match_id: selected.match_id, team: selected.team, venue: selected.venue})
      });
      if (!response.ok) throw new Error((await response.json()).detail || "Commit failed");
      location.reload();
    } catch (error) {
      document.querySelector("[data-commit-summary]").textContent = error.message;
      commitButton.disabled = false;
      commitButton.textContent = `Commit ${selected.team}`;
    }
  });

  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    if (event.key === "ArrowLeft") document.querySelector('.round-navigator a[aria-label="Previous week"]')?.click();
    if (event.key === "ArrowRight") document.querySelector('.round-navigator a[aria-label="Next week"]')?.click();
  });
  renderSelection();
})();
