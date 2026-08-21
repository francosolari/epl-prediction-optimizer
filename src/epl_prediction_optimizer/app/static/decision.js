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
  const commitButton = document.querySelector("[data-commit-button]");
  let selected = scenarios.find((item) => item.selected) || scenarios.find((item) => item.season_rank === 1);

  function pathFor(points) {
    if (!points.length) return "";
    const maxWeek = Math.max(...points.map((p) => p.week), 1);
    const maxValue = Math.max(...scenarios.flatMap((s) => (s.horizon || []).map((p) => p.value)), 1);
    return points.map((point, index) => {
      const x = 24 + (point.week / maxWeek) * 672;
      const y = 274 - (point.value / maxValue) * 238;
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  }

  function drawChart() {
    if (!pathsGroup || !gridGroup) return;
    gridGroup.innerHTML = [1, 2, 3, 4].map((i) => `<line x1="24" x2="696" y1="${i * 58}" y2="${i * 58}"/>`).join("");
    pathsGroup.innerHTML = scenarios.filter((s) => s.feasible).map((scenario) => {
      const active = selected && scenario.match_id === selected.match_id && scenario.team === selected.team;
      const kind = scenario.season_rank === 1 ? "season" : scenario.immediate_rank === 1 ? "now" : "alt";
      return `<path class="scenario-path ${kind} ${active ? "active" : ""}" d="${pathFor(scenario.horizon)}"><title>${scenario.team}: ${scenario.season_ev.toFixed(2)} projected expected points</title></path>`;
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
