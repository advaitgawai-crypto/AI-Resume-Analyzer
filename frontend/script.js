// ============================================================
// COUNTRY -> CITY DATA
// (JSearch supports these 19 countries. City lists are curated
//  major job hubs, not exhaustive — "Other" lets you type any city.)
// ============================================================
const COUNTRIES = [
  { code: "in", name: "India",          cities: ["Bangalore","Mumbai","Delhi","Hyderabad","Pune","Chennai","Kolkata","Gurgaon","Noida","Ahmedabad"] },
  { code: "us", name: "United States",  cities: ["New York","San Francisco","Seattle","Austin","Chicago","Boston","Los Angeles","Denver","Atlanta"] },
  { code: "gb", name: "United Kingdom", cities: ["London","Manchester","Birmingham","Edinburgh","Leeds","Bristol"] },
  { code: "ca", name: "Canada",         cities: ["Toronto","Vancouver","Montreal","Calgary","Ottawa","Edmonton"] },
  { code: "au", name: "Australia",      cities: ["Sydney","Melbourne","Brisbane","Perth","Adelaide","Canberra"] },
  { code: "de", name: "Germany",        cities: ["Berlin","Munich","Hamburg","Frankfurt","Cologne","Stuttgart"] },
  { code: "fr", name: "France",         cities: ["Paris","Lyon","Marseille","Toulouse","Bordeaux","Lille"] },
  { code: "nl", name: "Netherlands",    cities: ["Amsterdam","Rotterdam","The Hague","Utrecht","Eindhoven"] },
  { code: "it", name: "Italy",          cities: ["Milan","Rome","Turin","Bologna","Naples"] },
  { code: "es", name: "Spain",          cities: ["Madrid","Barcelona","Valencia","Seville"] },
  { code: "ch", name: "Switzerland",    cities: ["Zurich","Geneva","Basel","Bern"] },
  { code: "at", name: "Austria",        cities: ["Vienna","Graz","Linz","Salzburg"] },
  { code: "be", name: "Belgium",        cities: ["Brussels","Antwerp","Ghent","Leuven"] },
  { code: "br", name: "Brazil",         cities: ["São Paulo","Rio de Janeiro","Brasília","Belo Horizonte","Curitiba"] },
  { code: "mx", name: "Mexico",         cities: ["Mexico City","Guadalajara","Monterrey","Puebla"] },
  { code: "nz", name: "New Zealand",    cities: ["Auckland","Wellington","Christchurch"] },
  { code: "pl", name: "Poland",         cities: ["Warsaw","Krakow","Wroclaw","Poznan"] },
  { code: "sg", name: "Singapore",      cities: ["Singapore"] },
  { code: "za", name: "South Africa",   cities: ["Johannesburg","Cape Town","Pretoria","Durban"] },
];

// ============================================================
// STATE
// ============================================================
let lastReport = null; // holds the JSON returned by /api/analyze

// ============================================================
// DOM REFS
// ============================================================
const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("file-input");
const dropzoneFile   = document.getElementById("dropzone-file");
const uploadStatus   = document.getElementById("upload-status");

const stepReport     = document.getElementById("step-report");
const reportFilename = document.getElementById("report-filename");
const reportTime     = document.getElementById("report-time");
const reportScoreWidget = document.getElementById("report-score-widget");
const reportScoreText = document.getElementById("report-score-text");
const scoreRingFill  = document.getElementById("score-ring-fill");
const reportExperience = document.getElementById("report-experience");
const reportSkills   = document.getElementById("report-skills");
const reportTitlesBlock = document.getElementById("report-titles-block");
const reportTitles   = document.getElementById("report-titles");
const reportStrengths = document.getElementById("report-strengths");
const reportStrengthsNote = document.getElementById("report-strengths-note");
const reportDevelop  = document.getElementById("report-develop");
const actionPlanBlock = document.getElementById("action-plan-block");
const actionPlanDivider = document.getElementById("action-plan-divider");
const reportActionPlan = document.getElementById("report-action-plan");

const stepSearch     = document.getElementById("step-search");
const countrySelect  = document.getElementById("country-select");
const citySelect     = document.getElementById("city-select");
const cityManualField = document.getElementById("city-manual-field");
const cityManualInput = document.getElementById("city-manual");
const findJobsBtn    = document.getElementById("find-jobs-btn");
const searchStatus   = document.getElementById("search-status");
const jobsResults    = document.getElementById("jobs-results");

// ============================================================
// INIT
// ============================================================
function init() {
  populateCountries();
  bindUploadEvents();
  bindSearchEvents();
}

function populateCountries() {
  COUNTRIES.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.code;
    opt.textContent = c.name;
    countrySelect.appendChild(opt);
  });
}

// ============================================================
// UPLOAD HANDLING
// ============================================================
function bindUploadEvents() {
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });
  dropzone.setAttribute("tabindex", "0");
  dropzone.setAttribute("role", "button");
  dropzone.setAttribute("aria-label", "Upload job posting PDF");

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach(evt =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );

  ["dragleave", "drop"].forEach(evt =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });
}

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showStatus(uploadStatus, "Please upload a PDF file.", "error");
    return;
  }

  dropzone.classList.add("has-file");
  dropzoneFile.textContent = `📄 ${file.name}`;

  uploadFile(file);
}

async function uploadFile(file) {
  showStatus(uploadStatus, "Analyzing job posting — reading PDF, extracting skills, comparing against 2,716 resumes…", "loading");

  const formData = new FormData();
  formData.append("resume", file);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showStatus(uploadStatus, data.error || "Analysis failed.", "error");
      return;
    }

    lastReport = data;
    showStatus(uploadStatus, "Report ready.", "success");
    renderReport(data);
    stepReport.classList.remove("hidden");
    stepSearch.classList.remove("hidden");
    stepReport.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showStatus(uploadStatus, "Could not reach the server. Is app.py running?", "error");
  }
}

function showStatus(el, message, kind) {
  el.classList.remove("hidden", "error", "success");
  if (kind === "loading") {
    el.innerHTML = `<span class="spinner"></span><span>${message}</span>`;
  } else {
    el.innerHTML = `<span>${message}</span>`;
    if (kind) el.classList.add(kind);
  }
}

// ============================================================
// REPORT RENDERING
// ============================================================
function renderReport(data) {
  reportFilename.textContent = data.job_posting_name || "—";
  reportTime.textContent = data.analyzed_at || "";

  // Render Score
  const score = data.profile_score || 0;
  reportScoreText.textContent = score;
  
  // Set color class based on score
  reportScoreWidget.classList.remove("score-high", "score-med", "score-low");
  if (score >= 80) {
    reportScoreWidget.classList.add("score-high");
  } else if (score >= 50) {
    reportScoreWidget.classList.add("score-med");
  } else {
    reportScoreWidget.classList.add("score-low");
  }
  
  // Animate SVG ring (circumference is 100 for path length)
  setTimeout(() => {
    scoreRingFill.style.strokeDasharray = `${score}, 100`;
  }, 100);

  reportExperience.textContent = data.required_experience
    ? `${data.required_experience} years`
    : "Not specified";

  renderTags(reportSkills, data.core_skills);

  if (data.job_titles && data.job_titles.length) {
    reportTitlesBlock.classList.remove("hidden");
    renderTags(reportTitles, data.job_titles);
  } else {
    reportTitlesBlock.classList.add("hidden");
  }

  if (data.strengths && data.strengths.length) {
    renderTags(reportStrengths, data.strengths, "strength");
    reportStrengthsNote.textContent = data.strengths_note || "";
  } else {
    reportStrengths.innerHTML = `<span class="tag-empty">No overlapping skills found with top matches.</span>`;
    reportStrengthsNote.textContent = "";
  }

  renderSkillsToDevelop(data.skills_to_develop || []);

  if (data.action_plan) {
    actionPlanBlock.classList.remove("hidden");
    actionPlanDivider.classList.remove("hidden");
    reportActionPlan.textContent = data.action_plan;
  } else {
    actionPlanBlock.classList.add("hidden");
    actionPlanDivider.classList.add("hidden");
  }
}

function renderTags(container, items, variant) {
  container.innerHTML = "";
  if (!items || !items.length) {
    container.innerHTML = `<span class="tag-empty">None detected</span>`;
    return;
  }
  items.forEach(item => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = item;
    container.appendChild(tag);
  });
}

function renderSkillsToDevelop(skills) {
  reportDevelop.innerHTML = "";

  if (!skills.length) {
    reportDevelop.innerHTML = `<span class="tag-empty">No additional skills needed — strong match already.</span>`;
    return;
  }

  const priorityWidth = { HIGH: 90, MEDIUM: 55, LOW: 25 };

  skills.forEach(s => {
    const row = document.createElement("div");
    row.className = "skill-row";

    const name = document.createElement("div");
    name.className = "skill-row-name";
    name.textContent = s.skill;

    const priority = document.createElement("div");
    priority.className = `skill-row-priority priority-${s.priority}`;
    priority.textContent = s.priority;

    const meter = document.createElement("div");
    meter.className = "meter";
    const fill = document.createElement("div");
    fill.className = `meter-fill priority-fill-${s.priority}`;
    meter.appendChild(fill);

    row.appendChild(name);
    row.appendChild(priority);
    row.appendChild(meter);
    reportDevelop.appendChild(row);

    requestAnimationFrame(() => {
      fill.style.width = `${priorityWidth[s.priority] || 30}%`;
    });
  });
}

// ============================================================
// COUNTRY / CITY / JOB SEARCH
// ============================================================
function bindSearchEvents() {
  countrySelect.addEventListener("change", onCountryChange);
  citySelect.addEventListener("change", updateFindJobsState);
  cityManualInput.addEventListener("input", updateFindJobsState);
  findJobsBtn.addEventListener("click", runJobSearch);
}

function onCountryChange() {
  const code = countrySelect.value;
  citySelect.innerHTML = "";
  cityManualField.classList.add("hidden");
  cityManualInput.value = "";

  if (!code) {
    citySelect.disabled = true;
    citySelect.innerHTML = `<option value="">Select country first</option>`;
    updateFindJobsState();
    return;
  }

  const country = COUNTRIES.find(c => c.code === code);
  citySelect.disabled = false;

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select city";
  citySelect.appendChild(placeholder);

  country.cities.forEach(city => {
    const opt = document.createElement("option");
    opt.value = city;
    opt.textContent = city;
    citySelect.appendChild(opt);
  });

  const otherOpt = document.createElement("option");
  otherOpt.value = "__other__";
  otherOpt.textContent = "Other (type manually)";
  citySelect.appendChild(otherOpt);

  updateFindJobsState();
}

citySelect?.addEventListener("change", () => {
  if (citySelect.value === "__other__") {
    cityManualField.classList.remove("hidden");
  } else {
    cityManualField.classList.add("hidden");
    cityManualInput.value = "";
  }
});

function getSelectedCity() {
  if (citySelect.value === "__other__") return cityManualInput.value.trim();
  return citySelect.value;
}

function updateFindJobsState() {
  const hasCountry = !!countrySelect.value;
  const hasCity = !!getSelectedCity();
  findJobsBtn.disabled = !(hasCountry && hasCity && lastReport);
}

async function runJobSearch() {
  if (!lastReport) return;

  const countryCode = countrySelect.value;
  const city = getSelectedCity();
  const skills = lastReport.required_skills && lastReport.required_skills.length
    ? lastReport.required_skills
    : lastReport.core_skills;

  jobsResults.innerHTML = "";
  showStatus(searchStatus, `Searching for jobs in ${city}…`, "loading");
  findJobsBtn.disabled = true;

  try {
    const res = await fetch("/api/search-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skills, location: city, country_code: countryCode })
    });
    const data = await res.json();

    if (!res.ok) {
      showStatus(searchStatus, data.error || "Job search failed.", "error");
      findJobsBtn.disabled = false;
      return;
    }

    if (!data.jobs || !data.jobs.length) {
      showStatus(searchStatus, `No jobs found in ${city}. Try a different city.`, "error");
      findJobsBtn.disabled = false;
      return;
    }

    showStatus(searchStatus, `Found ${data.jobs.length} jobs in ${city}, ${data.country_name}.`, "success");
    renderJobs(data.jobs);

  } catch (err) {
    showStatus(searchStatus, "Could not reach the server.", "error");
  }

  findJobsBtn.disabled = false;
}

function renderJobs(jobs) {
  jobsResults.innerHTML = "";

  jobs.forEach(job => {
    const card = document.createElement("div");
    card.className = "job-card";

    const salaryText = (job.salary_min && job.salary_max)
      ? `<span class="salary">${job.salary_currency} ${Math.round(job.salary_min).toLocaleString()} – ${Math.round(job.salary_max).toLocaleString()}</span>`
      : "";

    const postedText = job.posting_date ? job.posting_date.split("T")[0] : "";

    card.innerHTML = `
      <div class="job-card-head">
        <div class="job-title">${escapeHtml(job.job_title)}</div>
        <div class="job-match">${job.match_score}% match</div>
      </div>
      <div class="job-company">${escapeHtml(job.company)} · ${escapeHtml(job.location)}</div>
      <div class="job-meter"><div class="job-meter-fill"></div></div>
      <div class="job-meta-row">
        ${salaryText}
        ${postedText ? `<span>posted ${postedText}</span>` : ""}
        <span>${job.matching_skills} skill(s) matched</span>
      </div>
      <div class="job-desc">${escapeHtml(job.description)}</div>
      <a class="job-apply" href="${job.application_url}" target="_blank" rel="noopener noreferrer">View & apply →</a>
    `;

    jobsResults.appendChild(card);

    const fill = card.querySelector(".job-meter-fill");
    requestAnimationFrame(() => {
      fill.style.width = `${job.match_score}%`;
    });
  });
}

function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ============================================================
init();