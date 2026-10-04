"""Build self-contained interactive dashboard for HLA-peptide stability dataset & splits."""
import json
from pathlib import Path
import pandas as pd

def build_dashboard():
    # Load allele statistics
    df_allele = pd.read_csv("splits/allele_statistics.csv")
    alleles_json = df_allele.to_json(orient="records")

    # Load metadata
    with open("splits/split_metadata.json", "r") as f:
        meta = json.load(f)
    meta_json = json.dumps(meta)

    # Load peptide top sample
    df_pep = pd.read_csv("splits/peptide_statistics.csv")
    pep_sample = df_pep.head(200).to_json(orient="records")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HLA-Peptide Stability & Generalization Splits Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    /* Custom scrollbars */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(150, 150, 150, 0.3); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(150, 150, 150, 0.5); }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-4 md:p-8 min-h-screen">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Header Section -->
    <header class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2 mb-1">
            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-500 border border-blue-500/20">Rasmussen et al. Dataset</span>
            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">5 Splitting Regimes</span>
            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-500/10 text-purple-500 border border-purple-500/20">Zero Leakage</span>
          </div>
          <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-[var(--foreground)]">HLA-Peptide Stability & ML Splits Explorer</h1>
          <p class="text-sm text-[var(--muted-foreground)] mt-1">
            Biophysical distribution profiling of 28,166 pHLA-I complexes across 75 allotypes and 5,633 peptides, featuring the 4-Quadrant Generalization Matrix and Ultra-Rare Allele ablations.
          </p>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-xs text-[var(--muted-foreground)] bg-[var(--background)] px-3 py-1.5 rounded-lg border border-[var(--border)] font-mono">
            t1/2 Threshold: &ge; 1.0 hour
          </span>
        </div>
      </div>

      <!-- KPI Summary Cards -->
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mt-6">
        <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs font-medium text-[var(--muted-foreground)]">Interactions</div>
          <div class="text-xl font-bold mt-0.5 text-[var(--foreground)]">28,166</div>
          <div class="text-[11px] text-blue-500 font-medium">100% 9-mer peptides</div>
        </div>
        <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs font-medium text-[var(--muted-foreground)]">HLA Allotypes</div>
          <div class="text-xl font-bold mt-0.5 text-[var(--foreground)]">75</div>
          <div class="text-[11px] text-indigo-500 font-medium">36 HLA-A &bull; 39 HLA-B</div>
        </div>
        <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs font-medium text-[var(--muted-foreground)]">Unique Peptides</div>
          <div class="text-xl font-bold mt-0.5 text-[var(--foreground)]">5,633</div>
          <div class="text-[11px] text-amber-500 font-medium">1,692 singletons (30%)</div>
        </div>
        <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs font-medium text-[var(--muted-foreground)]">Overall Binders</div>
          <div class="text-xl font-bold mt-0.5 text-emerald-500">51.93%</div>
          <div class="text-[11px] text-[var(--muted-foreground)]">14,626 / 28,166</div>
        </div>
        <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs font-medium text-[var(--muted-foreground)]">Ultra-Rare HLAs</div>
          <div class="text-xl font-bold mt-0.5 text-rose-500">7</div>
          <div class="text-[11px] text-rose-500 font-medium">N &lt; 50 (137 samples)</div>
        </div>
        <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs font-medium text-[var(--muted-foreground)]">Generalization Matrix</div>
          <div class="text-xl font-bold mt-0.5 text-purple-500">4 Quadrants</div>
          <div class="text-[11px] text-purple-500 font-medium">Double-unseen test</div>
        </div>
      </div>
    </header>

    <!-- Navigation Tabs -->
    <div class="flex items-center gap-2 border-b border-[var(--border)] pb-2 overflow-x-auto">
      <button onclick="switchTab('tab-splits')" id="btn-splits" class="tab-btn px-4 py-2 rounded-xl text-sm font-semibold transition-colors bg-[var(--card)] text-blue-500 border border-[var(--border)] shadow-sm">
        &bull; 4-Quadrant Splits Matrix
      </button>
      <button onclick="switchTab('tab-alleles')" id="btn-alleles" class="tab-btn px-4 py-2 rounded-xl text-sm font-semibold transition-colors text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        &bull; HLA Allotypes Explorer (75)
      </button>
      <button onclick="switchTab('tab-peptides')" id="btn-peptides" class="tab-btn px-4 py-2 rounded-xl text-sm font-semibold transition-colors text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        &bull; Peptides & Promiscuity
      </button>
      <button onclick="switchTab('tab-plots')" id="btn-plots" class="tab-btn px-4 py-2 rounded-xl text-sm font-semibold transition-colors text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        &bull; Publication Plots Gallery (8)
      </button>
      <button onclick="switchTab('tab-protocol')" id="btn-protocol" class="tab-btn px-4 py-2 rounded-xl text-sm font-semibold transition-colors text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        &bull; Foundation Model Protocol
      </button>
    </div>

    <!-- TAB 1: 4-Quadrant Splits Matrix -->
    <div id="tab-splits" class="tab-content space-y-6">
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h2 class="text-xl font-bold text-[var(--foreground)]">Biophysical Generalization Matrix (4 Quadrants)</h2>
            <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
              Decoupling sequence identity memorization from structural generalization across peptide and allele axes.
            </p>
          </div>
          <span class="text-xs font-semibold px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            Guaranteed Zero Peptide & Allele Leakage
          </span>
        </div>

        <!-- Interactive 2x2 Matrix Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Q1 -->
          <div class="p-5 rounded-2xl bg-emerald-500/5 border-2 border-emerald-500/30 hover:border-emerald-500 transition-all cursor-pointer">
            <div class="flex items-center justify-between">
              <span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-500 text-white">Quadrant 1</span>
              <span class="text-xs font-mono text-[var(--muted-foreground)]">quadrant_train_both_seen.csv</span>
            </div>
            <h3 class="text-lg font-bold text-[var(--foreground)] mt-2">Train Core (Seen HLA &times; Seen Peptide)</h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-1">Foundation corpus for representation learning and baseline in-distribution calibration.</p>
            <div class="grid grid-cols-3 gap-2 mt-4 text-center">
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Samples</div>
                <div class="text-sm font-bold text-[var(--foreground)]">14,238</div>
                <div class="text-[10px] text-emerald-500">50.6% total</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Alleles / Peps</div>
                <div class="text-sm font-bold text-[var(--foreground)]">49 / 3,572</div>
                <div class="text-[10px] text-[var(--muted-foreground)]">Core pool</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Binder Rate</div>
                <div class="text-sm font-bold text-emerald-500">53.2%</div>
                <div class="text-[10px] text-[var(--muted-foreground)]">Balanced</div>
              </div>
            </div>
          </div>

          <!-- Q2 -->
          <div class="p-5 rounded-2xl bg-blue-500/5 border-2 border-blue-500/30 hover:border-blue-500 transition-all cursor-pointer">
            <div class="flex items-center justify-between">
              <span class="px-2 py-0.5 rounded text-xs font-bold bg-blue-500 text-white">Quadrant 2</span>
              <span class="text-xs font-mono text-[var(--muted-foreground)]">quadrant_test_unseen_pep.csv</span>
            </div>
            <h3 class="text-lg font-bold text-[var(--foreground)] mt-2">Held-Out Peptides (Seen HLA &times; Unseen Peptide)</h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-1">Tests antigen / neoepitope discovery generalization on known patient allotypes.</p>
            <div class="grid grid-cols-3 gap-2 mt-4 text-center">
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Samples</div>
                <div class="text-sm font-bold text-[var(--foreground)]">2,990</div>
                <div class="text-[10px] text-blue-500">10.6% total</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Alleles / Peps</div>
                <div class="text-sm font-bold text-[var(--foreground)]">49 / 767</div>
                <div class="text-[10px] text-rose-500 font-semibold">0% in Train</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Binder Rate</div>
                <div class="text-sm font-bold text-blue-500">52.1%</div>
                <div class="text-[10px] text-[var(--muted-foreground)]">Balanced</div>
              </div>
            </div>
          </div>

          <!-- Q3 -->
          <div class="p-5 rounded-2xl bg-amber-500/5 border-2 border-amber-500/30 hover:border-amber-500 transition-all cursor-pointer">
            <div class="flex items-center justify-between">
              <span class="px-2 py-0.5 rounded text-xs font-bold bg-amber-500 text-white">Quadrant 3</span>
              <span class="text-xs font-mono text-[var(--muted-foreground)]">quadrant_test_unseen_allele.csv</span>
            </div>
            <h3 class="text-lg font-bold text-[var(--foreground)] mt-2">Held-Out Alleles (Unseen HLA &times; Seen Peptide)</h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-1">Tests pan-allele transfer to uncharacterized patient alleles using seen epitopes.</p>
            <div class="grid grid-cols-3 gap-2 mt-4 text-center">
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Samples</div>
                <div class="text-sm font-bold text-[var(--foreground)]">2,669</div>
                <div class="text-[10px] text-amber-500">9.5% total</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Alleles / Peps</div>
                <div class="text-sm font-bold text-[var(--foreground)]">9 / 2,013</div>
                <div class="text-[10px] text-rose-500 font-semibold">0% in Train</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Binder Rate</div>
                <div class="text-sm font-bold text-amber-500">50.6%</div>
                <div class="text-[10px] text-[var(--muted-foreground)]">Balanced</div>
              </div>
            </div>
          </div>

          <!-- Q4 -->
          <div class="p-5 rounded-2xl bg-rose-500/5 border-2 border-rose-500/30 hover:border-rose-500 transition-all cursor-pointer">
            <div class="flex items-center justify-between">
              <span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-500 text-white">Quadrant 4</span>
              <span class="text-xs font-mono text-[var(--muted-foreground)]">quadrant_test_double_unseen.csv</span>
            </div>
            <h3 class="text-lg font-bold text-[var(--foreground)] mt-2">Strict Double Held-Out (Unseen HLA &times; Unseen Peptide)</h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-1">The gold-standard biophysical generalization benchmark. Neither partner ever seen!</p>
            <div class="grid grid-cols-3 gap-2 mt-4 text-center">
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Samples</div>
                <div class="text-sm font-bold text-[var(--foreground)]">555</div>
                <div class="text-[10px] text-rose-500 font-semibold">Pure Benchmark</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Alleles / Peps</div>
                <div class="text-sm font-bold text-[var(--foreground)]">9 / 408</div>
                <div class="text-[10px] text-rose-500 font-semibold">Zero Overlap</div>
              </div>
              <div class="bg-[var(--background)] p-2 rounded-lg border border-[var(--border)]">
                <div class="text-[11px] text-[var(--muted-foreground)]">Binder Rate</div>
                <div class="text-sm font-bold text-rose-500">50.1%</div>
                <div class="text-[10px] text-[var(--muted-foreground)]">Perfect 50/50</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 5 Split Regimes Table -->
        <div class="mt-8">
          <h3 class="text-base font-bold text-[var(--foreground)] mb-3">Complete Suite of 5 Machine Learning Evaluation Regimes</h3>
          <div class="overflow-x-auto rounded-xl border border-[var(--border)]">
            <table class="w-full text-xs text-left">
              <thead class="bg-[var(--background)] border-b border-[var(--border)] text-[var(--muted-foreground)] font-semibold uppercase">
                <tr>
                  <th class="p-3">Regime</th>
                  <th class="p-3">Evaluation Focus</th>
                  <th class="p-3">Train Samples</th>
                  <th class="p-3">Val Samples</th>
                  <th class="p-3">Test Samples</th>
                  <th class="p-3">Test Binder %</th>
                  <th class="p-3">Generated Split Files</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[var(--border)]">
                <tr class="hover:bg-[var(--background)]/50">
                  <td class="p-3 font-semibold text-blue-500">1. IID Random</td>
                  <td class="p-3 text-[var(--muted-foreground)]">Standard in-distribution baseline (70/15/15)</td>
                  <td class="p-3 font-mono">19,716 (70%)</td>
                  <td class="p-3 font-mono">4,225 (15%)</td>
                  <td class="p-3 font-mono">4,225 (15%)</td>
                  <td class="p-3 font-bold text-emerald-500">51.9%</td>
                  <td class="p-3 font-mono text-[11px]">iid_train.csv, iid_val.csv, iid_test.csv</td>
                </tr>
                <tr class="hover:bg-[var(--background)]/50">
                  <td class="p-3 font-semibold text-cyan-500">2. Novel Peptides</td>
                  <td class="p-3 text-[var(--muted-foreground)]">Antigen / epitope holdout (zero peptide leakage)</td>
                  <td class="p-3 font-mono">19,848 (3,943 peps)</td>
                  <td class="p-3 font-mono">4,148 (845 peps)</td>
                  <td class="p-3 font-mono">4,170 (845 peps)</td>
                  <td class="p-3 font-bold text-emerald-500">50.8%</td>
                  <td class="p-3 font-mono text-[11px]">novel_pep_train.csv, novel_pep_test.csv</td>
                </tr>
                <tr class="hover:bg-[var(--background)]/50">
                  <td class="p-3 font-semibold text-amber-500">3. Novel Alleles</td>
                  <td class="p-3 text-[var(--muted-foreground)]">Pan-MHC transfer (zero allele leakage, 7 rare held out)</td>
                  <td class="p-3 font-mono">20,190 (49 HLAs)</td>
                  <td class="p-3 font-mono">4,066 (10 HLAs)</td>
                  <td class="p-3 font-mono">3,773 (9 HLAs)</td>
                  <td class="p-3 font-bold text-emerald-500">50.6%</td>
                  <td class="p-3 font-mono text-[11px]">novel_allele_train.csv, novel_allele_test.csv</td>
                </tr>
                <tr class="hover:bg-[var(--background)]/50">
                  <td class="p-3 font-semibold text-rose-500">4. Double Held-Out</td>
                  <td class="p-3 text-[var(--muted-foreground)]">4-Quadrant generalization matrix</td>
                  <td class="p-3 font-mono">14,238 (Q1 Core)</td>
                  <td class="p-3 font-mono">4,066 (Val Core)</td>
                  <td class="p-3 font-mono">555 (Q4 Double)</td>
                  <td class="p-3 font-bold text-emerald-500">50.1%</td>
                  <td class="p-3 font-mono text-[11px]">quadrant_*.csv (4 files)</td>
                </tr>
                <tr class="hover:bg-[var(--background)]/50">
                  <td class="p-3 font-semibold text-purple-500">5. Ultra-Rare Ablation</td>
                  <td class="p-3 text-[var(--muted-foreground)]">Few-shot (3-shot support) & zero-shot on N &lt; 50 HLAs</td>
                  <td class="p-3 font-mono">21 (3-shot)</td>
                  <td class="p-3 font-mono">-</td>
                  <td class="p-3 font-mono">116 (Query)</td>
                  <td class="p-3 font-bold text-emerald-500">50.9%</td>
                  <td class="p-3 font-mono text-[11px]">rare_3shot_support.csv, rare_heldout_query.csv</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: HLA Allotypes Explorer -->
    <div id="tab-alleles" class="tab-content hidden space-y-4">
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <h2 class="text-xl font-bold text-[var(--foreground)]">All 75 HLA Allotypes Profile</h2>
            <p class="text-xs text-[var(--muted-foreground)] mt-0.5">Filter by locus or rarity, search alleles, and inspect exact binding half-life distributions.</p>
          </div>
          <div class="flex flex-wrap items-center gap-2">
            <input type="text" id="allele-search" oninput="filterAlleles()" placeholder="Search HLA allele..." class="px-3 py-1.5 text-xs rounded-xl bg-[var(--background)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-blue-500 w-48">
            <select id="locus-filter" onchange="filterAlleles()" class="px-3 py-1.5 text-xs rounded-xl bg-[var(--background)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none">
              <option value="ALL">All Loci (75)</option>
              <option value="A">HLA-A (36)</option>
              <option value="B">HLA-B (39)</option>
            </select>
            <select id="rarity-filter" onchange="filterAlleles()" class="px-3 py-1.5 text-xs rounded-xl bg-[var(--background)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none">
              <option value="ALL">All Rarity Tiers</option>
              <option value="Ultra-Rare (<50)">Ultra-Rare (&lt; 50, 7 alleles)</option>
              <option value="Moderately-Rare (50-300)">Moderately Rare (50-300, 2 alleles)</option>
              <option value="Common (>=300)">Common (&ge; 300, 66 alleles)</option>
            </select>
          </div>
        </div>

        <!-- Allele Table -->
        <div class="overflow-x-auto rounded-xl border border-[var(--border)] max-h-[550px] overflow-y-auto">
          <table class="w-full text-xs text-left" id="allele-table">
            <thead class="bg-[var(--background)] sticky top-0 border-b border-[var(--border)] text-[var(--muted-foreground)] font-semibold uppercase">
              <tr>
                <th class="p-3 cursor-pointer" onclick="sortAlleles('allele')">Allele</th>
                <th class="p-3">Locus</th>
                <th class="p-3 cursor-pointer" onclick="sortAlleles('total_samples')">Samples</th>
                <th class="p-3 cursor-pointer" onclick="sortAlleles('pct_binder_1h')">Binder % (&ge;1h)</th>
                <th class="p-3">Binding Ratio Visual</th>
                <th class="p-3 cursor-pointer" onclick="sortAlleles('median_thalf')">Median t1/2</th>
                <th class="p-3">Rarity Tier</th>
                <th class="p-3">Pseudosequence (34 aa)</th>
              </tr>
            </thead>
            <tbody id="allele-tbody" class="divide-y divide-[var(--border)]">
              <!-- Dynamically populated via JS -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 3: Peptides & Promiscuity -->
    <div id="tab-peptides" class="tab-content hidden space-y-4">
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <h2 class="text-xl font-bold text-[var(--foreground)]">Peptide Distribution & Anchor Motifs</h2>
            <p class="text-xs text-[var(--muted-foreground)] mt-0.5">Explore 5,633 peptides, including 1,692 singletons, universal binders, and anchor residues.</p>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-xs text-amber-500 font-semibold px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20">
              1,692 Singletons (Tested 1x)
            </span>
            <span class="text-xs text-emerald-500 font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              1,388 Universal Binders (100%)
            </span>
          </div>
        </div>

        <!-- Promiscuity Summary Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-4 gap-3 mb-6">
          <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">Singletons (Tested 1x)</div>
            <div class="text-lg font-bold text-amber-500 mt-1">1,692 peptides</div>
            <div class="text-[11px] text-[var(--muted-foreground)]">30.0% of all peptides</div>
          </div>
          <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">Selective Binders</div>
            <div class="text-lg font-bold text-blue-500 mt-1">3,153 peptides</div>
            <div class="text-[11px] text-[var(--muted-foreground)]">56.0% (bind subset of HLAs)</div>
          </div>
          <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">Universal Binders</div>
            <div class="text-lg font-bold text-emerald-500 mt-1">1,388 peptides</div>
            <div class="text-[11px] text-[var(--muted-foreground)]">24.6% of peptides</div>
          </div>
          <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">Universal Non-Binders</div>
            <div class="text-lg font-bold text-rose-500 mt-1">1,092 peptides</div>
            <div class="text-[11px] text-[var(--muted-foreground)]">19.4% (never bind any HLA)</div>
          </div>
        </div>

        <!-- Anchor Motifs Callout -->
        <div class="p-4 rounded-xl bg-blue-500/5 border border-blue-500/20 text-xs space-y-2">
          <div class="font-bold text-blue-500">Biophysical Anchor Residue Insights (from Plot 6):</div>
          <p class="text-[var(--foreground)]">
            &bull; <strong>P2 (Pocket B):</strong> Leucine (L, +10.6% enrichment), Tyrosine (Y, +2.5%), and Threonine (T, +2.0%) strongly favored in binders. Proline (P, -4.7%) and Alanine (A, -3.2%) are depleted.<br>
            &bull; <strong>P9 (Pocket F, C-terminus):</strong> Valine (V, +6.3%), Tryptophan (W, +3.7%), and Arginine (R, +1.9%) show maximum binding enrichment.
          </p>
        </div>

        <!-- Sample Peptides Table -->
        <div class="mt-4">
          <h3 class="text-sm font-bold text-[var(--foreground)] mb-2">Sample Peptide Profiles (Top 200)</h3>
          <div class="overflow-x-auto rounded-xl border border-[var(--border)] max-h-[350px] overflow-y-auto">
            <table class="w-full text-xs text-left" id="peptide-table">
              <thead class="bg-[var(--background)] sticky top-0 border-b border-[var(--border)] text-[var(--muted-foreground)] font-semibold uppercase">
                <tr>
                  <th class="p-2.5">Peptide</th>
                  <th class="p-2.5">Tests</th>
                  <th class="p-2.5">Binders (&ge;1h)</th>
                  <th class="p-2.5">Binder Ratio</th>
                  <th class="p-2.5">Frequency Tier</th>
                  <th class="p-2.5">Profile</th>
                  <th class="p-2.5">P2 (Pocket B)</th>
                  <th class="p-2.5">P9 (Pocket F)</th>
                </tr>
              </thead>
              <tbody id="peptide-tbody" class="divide-y divide-[var(--border)]">
                <!-- Dynamically populated via JS -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: Publication Plots Gallery -->
    <div id="tab-plots" class="tab-content hidden space-y-6">
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-xl font-bold text-[var(--foreground)]">Generated Publication Figures (8 Plots, 300 DPI)</h2>
            <p class="text-xs text-[var(--muted-foreground)] mt-0.5">High-resolution scientific figures exported to <code>plots/</code>.</p>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Plot 1 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 1</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">01_hla_distribution_and_rarity.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">HLA Allele Distribution & Rarity Gap</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Distribution across all 75 HLA alleles showing the 7 ultra-rare alleles (N=7 to N=32) and the bimodal rarity gap between N=32 and N=220.
              </p>
            </div>
            <a href="plots/01_hla_distribution_and_rarity.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/01_hla_distribution_and_rarity.png" alt="Plot 1" class="w-full object-cover max-h-48">
            </a>
          </div>

          <!-- Plot 2 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 2</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">02_peptide_distribution_and_rarity.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">Peptide Distribution, Singletons & Promiscuity</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Singleton peak (1,692 peptides tested exactly once, 30.0%), empirical CDF, and promiscuity pie chart.
              </p>
            </div>
            <a href="plots/02_peptide_distribution_and_rarity.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/02_peptide_distribution_and_rarity.png" alt="Plot 2" class="w-full object-cover max-h-48">
            </a>
          </div>

          <!-- Plot 3 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 3</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">03_stability_and_binding_distribution.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">Stability & Binding Half-Life Distribution</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Bimodal log half-life distribution with 20.2% zero-spike, 5 stability tiers, and 51.9% binary balance.
              </p>
            </div>
            <a href="plots/03_stability_and_binding_distribution.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/03_stability_and_binding_distribution.png" alt="Plot 3" class="w-full object-cover max-h-48">
            </a>
          </div>

          <!-- Plot 4 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 4</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">04_allele_binding_proportions.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">Allele-Specific Binding Proportions</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Full 75-allele stacked bar chart showing stability variation from 3.2% binders in B*14:01 to 92.6% in A*02:11.
              </p>
            </div>
            <a href="plots/04_allele_binding_proportions.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/04_allele_binding_proportions.png" alt="Plot 4" class="w-full object-cover max-h-48">
            </a>
          </div>

          <!-- Plot 5 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 5</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">05_rare_entities_deepdive.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">Rare Entities Deep-Dive</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Ultra-rare HLAs binder breakdown, half-life boxplot comparison, and 98% peptide sharing with common HLAs.
              </p>
            </div>
            <a href="plots/05_rare_entities_deepdive.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/05_rare_entities_deepdive.png" alt="Plot 5" class="w-full object-cover max-h-48">
            </a>
          </div>

          <!-- Plot 6 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 6</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">06_peptide_anchor_motifs.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">Peptide Anchor Motifs (P2 & P9)</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Amino acid frequencies and enrichment deltas for Pocket B (P2) and Pocket F (P9).
              </p>
            </div>
            <a href="plots/06_peptide_anchor_motifs.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/06_peptide_anchor_motifs.png" alt="Plot 6" class="w-full object-cover max-h-48">
            </a>
          </div>

          <!-- Plot 7 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 7</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">07_split_regimes_matrix.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">4-Quadrant Generalization Matrix</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Comprehensive 2x2 diagram showing sample counts, unique entities, and binder balance across all quadrants.
              </p>
            </div>
            <a href="plots/07_split_regimes_matrix.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/07_split_regimes_matrix.png" alt="Plot 7" class="w-full object-cover max-h-48">
            </a>
          </div>

          <!-- Plot 8 -->
          <div class="border border-[var(--border)] rounded-xl p-4 bg-[var(--background)] flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-blue-500">Figure 8</span>
                <span class="text-[11px] text-[var(--muted-foreground)]">08_ablation_and_holdout_framework.png</span>
              </div>
              <h4 class="font-bold text-sm text-[var(--foreground)]">ML Ablation & Generalization Framework</h4>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 mb-3">
                Evaluation hierarchy sizes and expected foundation model generalization advantage on novel allotypes.
              </p>
            </div>
            <a href="plots/08_ablation_and_holdout_framework.png" target="_blank" class="block rounded-lg overflow-hidden border border-[var(--border)] hover:opacity-90">
              <img src="plots/08_ablation_and_holdout_framework.png" alt="Plot 8" class="w-full object-cover max-h-48">
            </a>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: Foundation Model Protocol -->
    <div id="tab-protocol" class="tab-content hidden space-y-6">
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-6">
        <div>
          <h2 class="text-xl font-bold text-[var(--foreground)]">Foundation Model (ESM-C & Boltz-2) Evaluation Protocol</h2>
          <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
            Step-by-step instructions to benchmark protein language models (ESM-C) and biomolecular complex predictors (Boltz-2).
          </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Model 1: ESM-C -->
          <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)] space-y-2">
            <span class="px-2 py-0.5 rounded text-xs font-bold bg-indigo-500 text-white">ESM-C (Evolutionary Scale Modeling)</span>
            <h4 class="text-sm font-bold text-[var(--foreground)]">Zero-Shot & Probing Head Strategy</h4>
            <p class="text-xs text-[var(--muted-foreground)]">
              1. Embed the 34-aa HLA pseudosequence and 9-mer peptide using ESM-C (e.g., <code>esmc_600m</code> or <code>esmc_300m</code>).<br>
              2. Compute representations by mean-pooling or taking the CLS embedding.<br>
              3. Concatenate <code>[h_allele; h_peptide; h_allele * h_peptide]</code>.<br>
              4. Train a lightweight MLP probing head strictly on <code>quadrant_train_both_seen.csv</code>.<br>
              5. Evaluate generalization drops across Q2, Q3, and Q4.
            </p>
          </div>

          <!-- Model 2: Boltz-2 -->
          <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)] space-y-2">
            <span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-500 text-white">Boltz-2 (Complex Structure Prediction)</span>
            <h4 class="text-sm font-bold text-[var(--foreground)]">Structural Confidence & Interface Scoring</h4>
            <p class="text-xs text-[var(--muted-foreground)]">
              1. Feed full-length alpha chain + beta-2 microglobulin + 9-mer peptide into Boltz-2.<br>
              2. Extract interface predicted TM-score (ipTM) and interface contact pLDDT.<br>
              3. Correlate structural confidence directly with continuous half-life (<code>thalf_hours</code>) via Spearman rank correlation.<br>
              4. Benchmark classification AUROC on the 555 double-unseen samples in Q4.
            </p>
          </div>
        </div>

        <!-- Python Code Loading Snippet -->
        <div>
          <h3 class="text-sm font-bold text-[var(--foreground)] mb-2">Loading Dataset & Splits in Python</h3>
          <pre class="bg-black/80 text-emerald-400 p-4 rounded-xl text-xs font-mono overflow-x-auto border border-emerald-500/20"><code>import pandas as pd

# 1. Master annotated dataset (all 28,166 pairs with split columns)
df = pd.read_csv("splits/dataset_with_splits.csv")

# 2. Extract 4-Quadrant Splits
train_q1 = df[df["split_quadrant"] == "quadrant_1_train_seen"]       # N=14,238
test_q2  = df[df["split_quadrant"] == "quadrant_2_test_unseen_pep"]  # N=2,990
test_q3  = df[df["split_quadrant"] == "quadrant_3_test_unseen_hla"]  # N=2,669
test_q4  = df[df["split_quadrant"] == "quadrant_4_test_double_unseen"]# N=555

# 3. Extract Ultra-Rare 3-shot few-shot benchmark
support_rare = df[df["split_rare_ablation"] == "rare_3shot_support"] # N=21
query_rare   = df[df["split_rare_ablation"] == "rare_heldout_query"] # N=116

print(f"Loaded Q1: {{len(train_q1)}}, Q2: {{len(test_q2)}}, Q3: {{len(test_q3)}}, Q4: {{len(test_q4)}}")
</code></pre>
        </div>
      </div>
    </div>

  </div>

  <script>
    const alleleData = {alleles_json};
    const peptideData = {pep_sample};
    const metadata = {meta_json};

    let currentAlleles = [...alleleData];
    let sortKey = 'total_samples';
    let sortAsc = false;

    function switchTab(tabId) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.remove('bg-[var(--card)]', 'text-blue-500', 'border', 'shadow-sm');
        btn.classList.add('text-[var(--muted-foreground)]');
      }});

      document.getElementById(tabId).classList.remove('hidden');
      const activeBtn = document.getElementById('btn-' + tabId.replace('tab-', ''));
      if (activeBtn) {{
        activeBtn.classList.remove('text-[var(--muted-foreground)]');
        activeBtn.classList.add('bg-[var(--card)]', 'text-blue-500', 'border', 'shadow-sm');
      }}
    }}

    function renderAlleles() {{
      const tbody = document.getElementById('allele-tbody');
      tbody.innerHTML = '';

      currentAlleles.forEach(a => {{
        const tr = document.createElement('tr');
        tr.className = "hover:bg-[var(--background)]/50 transition-colors";
        
        const isRare = a.total_samples < 50;
        const rarityBadge = isRare 
          ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-500 border border-rose-500/20">${{a.rarity_tier}}</span>`
          : (a.total_samples <= 300 
             ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/10 text-purple-500 border border-purple-500/20">${{a.rarity_tier}}</span>`
             : `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/10 text-blue-500 border border-blue-500/20">${{a.rarity_tier}}</span>`);

        tr.innerHTML = `
          <td class="p-3 font-mono font-bold ${{isRare ? 'text-rose-500' : 'text-[var(--foreground)]'}}">${{a.allele}}</td>
          <td class="p-3 font-semibold text-[var(--muted-foreground)]">${{a.locus}}</td>
          <td class="p-3 font-bold">${{a.total_samples.toLocaleString()}}</td>
          <td class="p-3 font-bold ${{a.pct_binder_1h >= 50 ? 'text-emerald-500' : 'text-amber-500'}}">${{a.pct_binder_1h.toFixed(1)}}%</td>
          <td class="p-3">
            <div class="w-24 bg-gray-200 dark:bg-gray-700 h-2 rounded-full overflow-hidden flex">
              <div style="width: ${{a.pct_binder_1h}}%" class="bg-emerald-500 h-full"></div>
              <div style="width: ${{100 - a.pct_binder_1h}}%" class="bg-gray-400 h-full"></div>
            </div>
          </td>
          <td class="p-3 font-mono">${{a.median_thalf.toFixed(1)}}h</td>
          <td class="p-3">${{rarityBadge}}</td>
          <td class="p-3 font-mono text-[10px] text-[var(--muted-foreground)] tracking-tighter">${{a.pseudoseq}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterAlleles() {{
      const query = document.getElementById('allele-search').value.toUpperCase();
      const locus = document.getElementById('locus-filter').value;
      const rarity = document.getElementById('rarity-filter').value;

      currentAlleles = alleleData.filter(a => {{
        const matchQ = a.allele.toUpperCase().includes(query);
        const matchL = locus === 'ALL' || a.locus === locus;
        const matchR = rarity === 'ALL' || a.rarity_tier === rarity;
        return matchQ && matchL && matchR;
      }});

      sortAlleles(sortKey, false);
    }}

    function sortAlleles(key, toggle = true) {{
      if (toggle) {{
        if (sortKey === key) sortAsc = !sortAsc;
        else {{ sortKey = key; sortAsc = false; }}
      }}

      currentAlleles.sort((a, b) => {{
        let vA = a[key], vB = b[key];
        if (typeof vA === 'string') {{
          return sortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
        }}
        return sortAsc ? vA - vB : vB - vA;
      }});
      renderAlleles();
    }}

    function renderPeptides() {{
      const tbody = document.getElementById('peptide-tbody');
      tbody.innerHTML = '';

      peptideData.forEach(p => {{
        const tr = document.createElement('tr');
        tr.className = "hover:bg-[var(--background)]/50 transition-colors";
        
        const isSingleton = p.total_tests === 1;
        const badge = p.binding_profile === 'Universal Binder'
          ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-500">Universal Binder</span>'
          : (p.binding_profile === 'Universal Non-Binder'
             ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-500">Non-Binder</span>'
             : '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/10 text-blue-500">Selective</span>');

        tr.innerHTML = `
          <td class="p-2.5 font-mono font-bold text-[var(--foreground)]">${{p.peptide}}</td>
          <td class="p-2.5 font-bold">${{p.total_tests}}</td>
          <td class="p-2.5">${{p.binder_1h_count}}</td>
          <td class="p-2.5 font-bold">${{(p.binder_ratio * 100).toFixed(0)}}%</td>
          <td class="p-2.5 text-[11px] text-[var(--muted-foreground)]">${{p.freq_tier}}</td>
          <td class="p-2.5">${{badge}}</td>
          <td class="p-2.5 font-mono font-semibold text-blue-500">${{p.p2_pocket_b}}</td>
          <td class="p-2.5 font-mono font-semibold text-indigo-500">${{p.p9_pocket_f}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // Initial render
    renderAlleles();
    renderPeptides();
  </script>
</body>
</html>
"""

    out_path = Path("/Users/mic/.gemini/antigravity/brain/df453d51-acfa-4b7c-baa7-474dd346ffda/dataset_splits_dashboard.html")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Dashboard successfully generated at {out_path} ({len(html_content):,} bytes)")

if __name__ == "__main__":
    build_dashboard()
