// ---- state ----
const el = (id) => document.getElementById(id);
let RACES = {};
let loadingRace = false;   // guard: suppress refresh while we set controls programmatically

// ---- init ----
init();
async function init() {
  const [meta, races] = await Promise.all([
    fetch("/api/meta").then((r) => r.json()),
    fetch("/api/races").then((r) => r.json()),
  ]);
  renderMetrics(meta.metrics, meta.n_rows, meta.seasons.length);
  RACES = races.races;

  const seasonSel = el("season");
  Object.keys(RACES).sort((a, b) => b - a).forEach((s) => {
    seasonSel.add(new Option(s, s));
  });
  seasonSel.onchange = fillCircuits;
  el("circuit").onchange = loadRace;

  // control listeners
  el("rainBtn").onclick = () => { toggleRain(); refresh(); };
  el("trackTemp").oninput = () => { el("trackTempVal").textContent = el("trackTemp").value + "°C"; refresh(); };
  el("wind").oninput = () => { el("windVal").textContent = el("wind").value; refresh(); };
  el("penaltyDriver").onchange = refresh;
  el("penaltyGrid").oninput = () => { el("penaltyGridVal").textContent = el("penaltyGrid").value; refresh(); };
  el("liveBtn").onclick = useLiveWeather;
  el("resetBtn").onclick = () => { el("liveStatus").textContent = ""; loadRace(); };

  fillCircuits();
}

function fillCircuits() {
  const s = el("season").value;
  const c = el("circuit");
  c.innerHTML = "";
  RACES[s].forEach((r) => c.add(new Option(r, r)));
  loadRace();
}

// Load a race fresh: pull actual conditions, reset controls to them, render.
async function loadRace() {
  loadingRace = true;
  const season = el("season").value;
  const circuit = el("circuit").value;
  el("penaltyDriver").value = "";
  const data = await callPredict({ season, circuit, rain: "actual" });
  if (!data) { loadingRace = false; return; }

  // sync controls to the race's real conditions
  setRain(data.weather.actual_rain === 1);
  el("trackTemp").value = data.weather.track_temp ?? 30;
  el("trackTempVal").textContent = (data.weather.track_temp ?? 30) + "°C";
  el("wind").value = data.weather.wind_speed ?? 1;
  el("windVal").textContent = data.weather.wind_speed ?? 1;

  // populate penalty-driver options
  const pd = el("penaltyDriver");
  pd.innerHTML = '<option value="">No penalty</option>';
  data.drivers
    .slice()
    .sort((a, b) => (a.grid ?? 99) - (b.grid ?? 99))
    .forEach((d) => pd.add(new Option(`${d.name} (${d.driver})`, d.driver)));

  loadingRace = false;
  render(data);
}

// Re-predict using current control state (what-if scenario).
async function refresh() {
  if (loadingRace) return;
  const data = await callPredict({
    season: el("season").value,
    circuit: el("circuit").value,
    rain: el("rainBtn").classList.contains("wet") ? "1" : "0",
    track_temp: el("trackTemp").value,
    wind: el("wind").value,
    penalty_driver: el("penaltyDriver").value,
    penalty_grid: el("penaltyGrid").value,
  });
  if (data) render(data);
}

async function callPredict(params) {
  el("loader").classList.remove("hidden");
  try {
    const q = new URLSearchParams(params).toString();
    const res = await fetch(`/api/predict?${q}`);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  } finally {
    el("loader").classList.add("hidden");
  }
}

// ---- live weather ----
async function useLiveWeather() {
  const circuit = el("circuit").value;
  const status = el("liveStatus");
  status.textContent = "fetching…";
  try {
    const w = await fetch(`/api/weather/live?circuit=${encodeURIComponent(circuit)}`).then((r) => r.json());
    if (w.source !== "live") {
      status.textContent = (w.reason || "").toLowerCase().includes("key")
        ? "Set OPENWEATHER_API_KEY to enable live weather (see README)."
        : `Live weather unavailable (${w.reason || "error"}).`;
      return;
    }
    setRain(w.rainfall === 1);
    el("trackTemp").value = w.track_temp;
    el("trackTempVal").textContent = w.track_temp + "°C";
    el("wind").value = w.wind_speed;
    el("windVal").textContent = w.wind_speed;
    status.textContent = `Live @ ${w.location}: ${w.air_temp}°C air · ${w.condition}`;
    refresh();
  } catch (e) {
    status.textContent = "Live weather request failed.";
  }
}

// ---- rain toggle helpers ----
function setRain(wet) {
  const b = el("rainBtn");
  b.classList.toggle("wet", wet);
  b.textContent = wet ? "Wet 🌧️" : "Dry ☀️";
}
function toggleRain() { setRain(!el("rainBtn").classList.contains("wet")); }

// ---- rendering ----
function renderMetrics(m, nRows, nSeasons) {
  const lift = (m.roc_auc - m.grid_baseline_auc).toFixed(3);
  el("metrics").innerHTML = `
    <div class="badge"><div class="k">ROC-AUC</div><div class="v">${m.roc_auc.toFixed(3)}</div><div class="sub">▲ +${lift} vs grid</div></div>
    <div class="badge"><div class="k">Precision@3</div><div class="v">${m.precision_at_3.toFixed(2)}</div><div class="sub">podium hit-rate</div></div>
    <div class="badge"><div class="k">Seasons</div><div class="v">${nSeasons}</div><div class="sub">${nRows} driver-races</div></div>`;
  el("honest").textContent =
    `Honest note: the starting grid alone is a very strong predictor, so the model only ~ties it on the exact top-3 ` +
    `(Precision@3 ${m.precision_at_3.toFixed(2)} vs ${m.grid_baseline_p3.toFixed(2)}), but it beats grid on full-field ranking ` +
    `(AUC ${m.roc_auc.toFixed(3)} vs ${m.grid_baseline_auc.toFixed(3)}) by also reading pace, form, reliability and weather.`;
}

function pct(p) { return Math.round((p ?? 0) * 100); }

function render(data) {
  renderWeather(data.weather);
  renderBanner(data);
  renderPodium(data);
  renderDrivers(data);
  renderConstructors(data.constructors);
}

function renderWeather(w) {
  const wet = w.rainfall === 1;
  el("weatherStrip").innerHTML = `
    <div class="w">🌡️ <b>${w.track_temp ?? "?"}°C</b> <small>track</small></div>
    <div class="w">🌤️ <b>${w.air_temp ?? "?"}°C</b> <small>air</small></div>
    <div class="w">💧 <b>${w.humidity ?? "?"}%</b> <small>humidity</small></div>
    <div class="w">💨 <b>${w.wind_speed ?? "?"}</b> <small>wind</small></div>
    <div class="w">${wet ? "🌧️ <b>Wet</b>" : "☀️ <b>Dry</b>"} <small>conditions</small></div>`;
}

function renderBanner(data) {
  const b = el("banner");
  if (data.race.changed) {
    b.innerHTML = `<span class="box whatif">⚙️ What-if scenario — you changed the conditions, so there's no actual result to compare against.</span>`;
  } else if (data.actual_podium.length) {
    b.innerHTML = `<span class="box hit">✅ Model got <b>${data.hits}/3</b> of the actual podium — real result: ${data.actual_podium.join(", ")}.</span>`;
  } else {
    b.innerHTML = "";
  }
}

function renderPodium(data) {
  const medals = ["🥇", "🥈", "🥉"];
  const showHit = !data.race.changed && data.actual_podium.length;
  el("podium").innerHTML = data.drivers.slice(0, 3).map((d, i) => {
    let mark = "";
    if (showHit) {
      mark = d.podium_hit
        ? `<div class="mark ok">✅ finished on the podium</div>`
        : `<div class="mark no">✕ did not podium</div>`;
    }
    return `
      <div class="pcard rank${i + 1}" style="--tc:${d.team_color}">
        <div class="pos">${medals[i]} P${i + 1}</div>
        <div class="dname">${d.name}</div>
        <div class="team">${d.team} · started P${d.grid ?? "?"}</div>
        <div class="prob">${pct(d.podium_proba)}<small>% podium chance</small></div>
        <div class="bar"><i style="width:${pct(d.podium_proba)}%"></i></div>
        ${mark}
      </div>`;
  }).join("");
}

function renderDrivers(data) {
  const podSet = new Set(data.predicted_podium);
  el("drivers").innerHTML = data.drivers.map((d) => {
    const fin = d.actual_finish
      ? `<b>P${d.actual_finish}</b><small>actual</small>`
      : `<small>—</small>`;
    return `
      <div class="drow ${podSet.has(d.driver) ? "pod" : ""}" style="--tc:${d.team_color}">
        <div class="rk">${d.pred_rank}</div>
        <div class="who"><b>${d.name}</b><small>${d.team} · grid P${d.grid ?? "?"}</small></div>
        <div class="pbar"><i style="width:${pct(d.podium_proba)}%"></i><span>${pct(d.podium_proba)}%</span></div>
        <div class="fin">${fin}</div>
      </div>`;
  }).join("");
}

function renderConstructors(cons) {
  const max = Math.max(...cons.map((c) => c.pred_points), 1);
  el("constructors").innerHTML = cons.map((c) => `
    <div class="crow">
      <div class="top"><b>${c.team}</b><span>${c.pred_points} pts</span></div>
      <div class="cbar"><i style="width:${(c.pred_points / max) * 100}%; background:${c.team_color}"></i></div>
    </div>`).join("");
}
