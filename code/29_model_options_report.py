#!/usr/bin/env python3
"""29_model_options_report.py -- surrogate-model prediction-task selection report (candidate recommendations, pending the user's choice).

Reads only existing metadata (manifest / run_status / data_guide / batch summary) and the results of this literature check,
and produces result/reports/REPORT_CURRENT.pdf. Does not read netCDF files in runs/; does not modify data/ or runs/.
"""
import os, json
import pandas as pd
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'Songti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'result', 'reports', 'REPORT_CURRENT.pdf')
EVS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']

S = json.load(open(os.path.join(ROOT, 'data', 'static', 'meta', 'batch486_summary_2026-09-16.json')))
G = json.load(open(os.path.join(ROOT, 'data', 'static', 'meta', 'data_guide_486.json')))
MAN = {e: pd.read_csv(os.path.join(ROOT, 'data', e, 'scenarios', 'manifest.csv')) for e in EVS}

# ---------- layout helpers ----------
CW = lambda ch: 1.0 if ord(ch) > 0x2e80 else 0.52          # CJK full width / Latin half width


def wrap(text, width):
    """Wrap by visual character width; width is in units of full-width characters."""
    out, cur, w = [], '', 0.0
    for ch in str(text):
        if ch == '\n':
            out.append(cur); cur, w = '', 0.0; continue
        c = CW(ch)
        if w + c > width and cur:
            out.append(cur); cur, w = '', 0.0
        cur += ch; w += c
    out.append(cur)
    return out


BOTTOM = 0.055


class Doc:
    """A4 portrait text + table pages with automatic page breaks."""
    def __init__(self, pdf):
        self.pdf, self.fig, self.y = pdf, None, None
        self.page = 0
        self.title = None

    def new(self, title=None, cont=False):
        if self.fig is not None:
            self.pdf.savefig(self.fig); plt.close(self.fig)
        self.fig = plt.figure(figsize=(8.27, 11.69)); self.page += 1
        if title is not None:
            self.title = title
        self.y = 0.955
        t = (self.title + ' (cont.)') if (cont and self.title) else title
        if t:
            self.fig.text(0.07, 0.962, t, fontsize=13, weight='bold'); self.y = 0.925
        self.fig.text(0.93, 0.024, 'Page %d' % self.page, fontsize=7.5, color='#666', ha='right')
        return self

    def ensure(self, h):
        """Ensure h (figure fraction) of height is still available, otherwise start a new page."""
        if self.y - h < BOTTOM:
            self.new(cont=True)
        return self

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig); plt.close(self.fig); self.fig = None

    def text(self, s, fs=8.5, dy=0.0152, indent=0.0, weight=None, color='black', width=None):
        w = width or (0.86 * 8.27 * 72 / fs * 0.97)
        for line in wrap(s, w):
            self.ensure(dy)
            self.fig.text(0.07 + indent, self.y, line, fontsize=fs, va='top', weight=weight, color=color)
            self.y -= dy
        return self

    def gap(self, dy=0.008):
        self.y -= dy; return self

    def rule(self):
        self.fig.add_artist(plt.Line2D([0.07, 0.93], [self.y + 0.006, self.y + 0.006],
                                       color='#bbb', lw=0.6, transform=self.fig.transFigure))
        self.y -= 0.008; return self

    def table(self, header, rows, colw, fs=6.9, lh=0.0118, pad=0.004):
        """colw = column widths (figure fractions, sum <= 0.86)."""
        x0 = 0.07
        xs, acc = [], x0
        for c in colw:
            xs.append(acc); acc += c
        chars = [c * 8.27 * 72 / fs for c in colw]   # full-width characters that fit in each column

        def draw_row(cells, bold=False, bg=None):
            wrapped = [wrap(c, max(chars[i] - 0.8, 3)) for i, c in enumerate(cells)]
            h = max(len(w) for w in wrapped) * lh + pad
            self.ensure(h)
            if bg:
                self.fig.patches.append(plt.Rectangle((x0, self.y - h + lh * 0.55), acc - x0, h,
                                                      transform=self.fig.transFigure, facecolor=bg,
                                                      edgecolor='none', zorder=0))
            for i, cell in enumerate(wrapped):
                yy = self.y
                for line in cell:
                    self.fig.text(xs[i] + 0.004, yy, line, fontsize=fs, va='top',
                                  weight='bold' if bold else None, zorder=2)
                    yy -= lh
            self.y -= h
            self.fig.add_artist(plt.Line2D([x0, acc], [self.y + lh * 0.55] * 2, color='#ccc', lw=0.5,
                                           transform=self.fig.transFigure))

        draw_row(header, bold=True, bg='#e9e9e9')
        for r in rows:
            draw_row(r)
        self.y -= 0.006
        return self


# ---------- data facts (all from verified metadata) ----------
n_run = S['n']
frames_all = n_run * G['ampr']['n'][0]
frames_evt = n_run * 168
wet = {e: (S['per_event'][e]['wet_min'] * 100, S['per_event'][e]['wet_max'] * 100, S['per_event'][e]['wet_s041'] * 100) for e in EVS}
rain = {e: (MAN[e].rain_total_mm.min(), MAN[e].rain_total_mm.max()) for e in EVS}

with PdfPages(OUT) as pdf:
    d = Doc(pdf)

    # ===================== page 1 =====================
    d.new('Surrogate-model prediction task selection -- candidate recommendations, pending user choice')
    d.text('2026-09-16. Subject: the 486 completed and verified SFINCS simulations (six hurricanes x 81 Irma-like compound forcing scenarios). '
           'This report only surveys options and makes recommendations; no data was packaged, no training split was made, no model was trained, and Proposal.md / data/ / runs/ were not modified.', fs=8.2)
    d.gap()

    d.text('1. Conclusions first', fs=11, weight='bold'); d.rule()
    for s in [
        '1. With the existing data, the most robust and quickest task is "scenario forcing -> 7-day main-period max water depth / inundation extent" (Option A). '
        'Each simulation contributes 1 sample, 486 in total; the target maps already exist as event7d/hmax_event7d_hourlysampled.npy in each run directory and need no recomputation.',
        '2. Hourly depth fields (Option B) and autoregressive rollout (Option C) are both feasible but rank below A: the 81.6k main-period frames of B are highly correlated, '
        'and C also requires known future hourly rainfall / boundary water level / discharge while stepping forward, and accumulates error. Recommended as stages two and three.',
        '3. The six real events are a hard ceiling for generalization testing. Cross-event testing allows only 6-fold leave-one-out, and the events differ greatly ('
        'total rainfall %.0f-%.0f mm, 7-day land inundation fraction %.0f-%.0f %%), so leave-one-out is often extrapolation rather than interpolation; conclusions must be reported per fold.'
        % (min(r[0] for r in rain.values()), max(r[1] for r in rain.values()),
           min(w[0] for w in wet.values()), max(w[1] for w in wet.values())),
        '4. Recommended order: A (scenario -> max depth) -> D (cross-event leave-one-out protocol) -> E (few-shot fine-tuning) -> B (direct per-frame prediction) -> C (autoregressive). '
        'A + D + E alone can support a complete paper, all using existing data only.',
        '5. Preferred model: per-cell random forest / gradient boosting (Option A, interpretable, runnable within a day); alternative U-Net (image-to-image, a baseline for A and the main model for B). '
        'FNO is suggested for stage B / C.',
    ]:
        d.text('• ' + s, fs=8.4); d.gap(0.003)
    d.gap()

    d.text('2. Comparison of candidate options', fs=11, weight='bold'); d.rule()
    d.table(
        ['Option', 'Predicts', 'Input', 'Output', 'Needs future forcing', 'Preferred / alt. model', 'Effort', 'Rating'],
        [
            ['A event max depth / inundation extent',
             'per-cell max depth and inundation extent over the whole event (7-day main period)',
             '4 scenario factors α β γ τ + whole-event forcing summary scalars + per-cell static features (zb, mask, location)',
             'per-cell h_max (m); h>0.1 / 0.3 m binary extent',
             'needs whole-event forcing (design / scenario assessment, not forecasting)',
             'per-cell random forest / gradient boosting; U-Net',
             'small',
             '★★★★★'],
            ['B hourly depth field (direct)',
             'domain-wide depth field at a given hour, no recursion',
             'static layers + rainfall field / boundary water level / discharge at the current and several previous hours + time encoding',
             'h field at that hour (195x255)',
             'no, uses current and past only -> quasi-forecasting possible',
             'U-Net; FNO',
             'medium',
             '★★★★'],
            ['C autoregressive rollout',
             'recursively predict h(t+1) from h(t), rolling out to 168 h',
             'h(t) and several previous steps + forcing increment t->t+1 + static layers',
             'hourly depth sequence',
             'yes: rollout requires future hourly rain / water level / discharge',
             'U-Net / ConvLSTM single-step + curriculum multi-step training; GNN',
             'large',
             '★★'],
            ['D cross-event leave-one-out',
             'not a new model but an evaluation protocol: one hurricane fully excluded from training and tuning',
             'same as A or B',
             '6-fold leave-one-out error and CSI, reported per fold',
             'same as chosen option',
             'reuse the A / B model',
             'small (on top of A)',
             '★★★★★'],
            ['E few-shot fine-tuning',
             'how much adapting with only 1 / 3 / 5 scenarios of the target hurricane helps',
             'source-model weights + a few scenarios of the target event',
             'three comparison curves: zero-shot / fine-tuned / trained from scratch on the same data',
             'same as chosen option',
             'first retrain the tree model, then fine-tune U-Net with some layers frozen',
             'medium (on top of A + D)',
             '★★★★'],
            ['F gauge water-level time series',
             'water level / depth time series at the 7 in-domain stations in sfincs_his.nc',
             'forcing time series (lag terms possible)',
             '10-min or hourly water level per station',
             'depends on setup; forecasting needs future forcing',
             'gradient boosting / LSTM',
             'very small',
             '★★ (supplementary)'],
        ],
        colw=[0.105, 0.125, 0.185, 0.115, 0.115, 0.125, 0.048, 0.042])

    # ===================== page 2 =====================
    d.new('3. Implementation outline per option (outline only; no network architecture or training parameters)')
    outlines = [
        ('A  Scenario forcing -> 7-day main-period max depth / inundation extent', [
            'What it predicts / use: given a set of compound forcings (tidal residual, rainfall, low-frequency river flow, rainfall time shift), return the whole-event max depth map and inundation extent within seconds. '
            'Uses: scenario screening, probabilistic flood assessment, sensitivity analysis -- the same uses as SFINCS ensemble simulation.',
            'Input -> output: input = the four factors α β γ τ + summary scalars of the scenario\'s whole-event forcing (total rainfall, boundary water-level peak and peak time, Q_low peak, etc.) + per-cell static features; '
            'output = per-cell max depth, and binary inundation maps at the 0.1 / 0.3 m thresholds.',
            'Use of existing data: the target maps already exist -- event7d/hmax_event7d_hourlysampled.npy in each run directory is the 7-day main period (t = 73...240 h) '
            'per-cell hourly-sampled max depth. 486 simulations = 486 samples, 81 per event. Forcing summary scalars can be computed directly from manifest.csv and bzs / dis / ampr.',
            'Preferred / alternative: preferred per-cell random forest or gradient boosting (Zahura 2020 trained a point-wise RF per road segment as a TUFLOW surrogate; interpretable, feature importance directly readable); '
            'alternative U-Net, broadcasting the forcing scalars as channels concatenated with static layers (Löwe 2021 U-FLOOD is exactly "terrain + rainfall pattern -> max depth map").',
            'Training and validation: experiment A uses the frozen split_A (60 / 11 / 10 per event, split by whole scenarios, seed 42); '
            'metrics = wet-cell MAE / RMSE, inundated-area error, CSI (reported at both 0.1 and 0.3 m).',
            'Effort / risk / recommendation: smallest effort (486 samples, low feature dimension, runs on one machine); '
            'risk = the 81 scenarios within one event differ only by 4 scalars, so the model may just learn 4-D interpolation, which must be exposed by the cross-event folds of Option D; highest recommendation, suggested as the starting point.',
        ]),
        ('B  Forcing + time -> depth field at that hour (direct prediction, no recursion)', [
            'What it predicts / use: the domain-wide depth field at any given hour. Supports quasi-forecasting ("given forecast forcing for the next few hours -> current and near-term inundation"), and serves as an upper-bound reference for C.',
            'Input -> output: input = static layers (zb, mask) + rainfall field, boundary water level and upstream discharge at the current and several previous hours + time encoding; output = h field at that hour.',
            'Use of existing data: 241 hourly snapshots per simulation, 168 per simulation after removing the 72 h spin-up, %s main-period frames over the six events. '
            'Splits must be by whole scenarios, not random by frame.' % format(frames_evt, ','),
            'Preferred / alternative: preferred U-Net per-frame image-to-image; alternative FNO. All six events share the same 200 m grid and terrain, so no cross-terrain generalization is needed, which helps both.',
            'Training and validation: reuse split_A; report hourly error over time and error at peak time.',
            'Effort / risk / recommendation: medium effort; risk = 80k frames are not 80k independent samples (adjacent hours within a scenario are highly autocorrelated, '
            'and scenarios within an event differ by only 4 scalars); a random per-frame split would give inflated accuracy; recommended as the second stage after A.',
        ]),
        ('C  Autoregressive rollout (h(t) -> h(t+1), rolling out 168 h)', [
            'What it predicts / use: steps the whole event forward hour by hour from an initial field; closest to a "replacement solver" and the easiest framing for a physics-ML story.',
            'Input -> output: input = h(t) and several previous steps + forcing for t->t+1 + static layers; output = h(t+1), with the prediction fed back as the next input.',
            'Needs future forcing: yes. Rolling to step k requires rainfall, boundary water level and discharge at hour k; this must be stated clearly in the paper, otherwise it is simulation acceleration rather than forecasting.',
            'Use of existing data: the hourly snapshots were kept for exactly this purpose (dtout = 3600, dtmaxout = 0, 481 runs without zsmax). '
            'Training starts with a single-step loss, then gradually lengthens the prediction horizon following the curriculum strategy of Bentivoglio 2023.',
            'Preferred / alternative: preferred U-Net or ConvLSTM single-step + curriculum multi-step; alternative graph neural network (SWE-GNN) or FNO.',
            'Effort / risk / recommendation: largest effort; high risk -- error accumulation, wet/dry switching (h <= 0.05 m is dry), no mass-conservation constraint; '
            'long rollouts in the literature all need dedicated stabilization techniques. Suggested last, or only as an extension after A / B succeed.',
        ]),
    ]
    for title, items in outlines:
        d.text(title, fs=9.4, weight='bold'); d.gap(0.002)
        for it in items:
            d.text('· ' + it, fs=8.1, dy=0.0143, indent=0.008)
        d.gap(0.006)

    # ===================== page 3 =====================
    d.new('3 (cont.) Cross-event and fine-tuning; 4. Proposal check; 5. Decision recommendations')
    for title, items in [
        ('D  Cross-event generalization test (an evaluation protocol, not a separate model)', [
            'What: hold out one hurricane, with all its 81 scenarios excluded from training and tuning and used only for final testing; rotating over the six events = 6 folds. '
            'A fixed-budget comparison is also possible (e.g. "1 event x 50 scenarios" vs "5 events x 10 scenarios"), by subsampling the 486 runs; no new simulations needed.',
            'Limitation that must be reported honestly: the forcing ranges of the six events differ greatly ('
            + '; '.join('%s rain %.0f-%.0f mm, inundation %.0f-%.0f %%' % (e, rain[e][0], rain[e][1], wet[e][0], wet[e][1]) for e in ['irma', 'ian'])
            + ', etc.), so the held-out event often falls outside the training range, i.e. extrapolation; each fold must state whether it is interpolation or extrapolation. '
              'Feng 2025 observed clear U-Net degradation on an unseen hurricane while a CNN-LSTM with temporal structure was more robust, so this risk has a literature precedent.',
            'Small effort (built on A), recommendation tied with A as highest -- this is the project\'s real contribution relative to existing literature.',
        ]),
        ('E  Few-shot fine-tuning (corresponds to Proposal experiment C)', [
            'What: source model = trained on the other five events; adapt to the target hurricane with only 1 / 3 / 5 scenarios; compare against two baselines, "zero-shot" and "trained from scratch on the same amount of data".',
            'Use of existing data: among the 81 scenarios of the target event, the adaptation pool and the final test scenarios are drawn separately; test scenarios are not used for adaptation or tuning.',
            'Evidence: the fine-tuned deep neural operator of Xu 2025 works with very few labelled target-domain samples (error drops clearly once labelled target samples reach 4 or more); '
            'Seleem 2023 found that CNNs benefit more from transfer learning than random forests -- so the tree model is suggested only as a retraining baseline, with fine-tuning experiments on the U-Net.',
            'Medium effort, high recommendation: it turns the weakness "only 6 events" into a selling point of the paper (how much simulation budget a new hurricane requires).',
        ]),
        ('F  Gauge water-level time series (supplementary, not the main line)', [
            'Each simulation\'s sfincs_his.nc contains 9 stations and 1441 10-min time steps, of which 7 stations are inside the domain. '
            'Gradient boosting or LSTM could do "forcing -> station water level" at very low cost; suitable as an auxiliary validation figure in an Option A paper, not recommended as the main task.',
        ]),
    ]:
        d.text(title, fs=9.4, weight='bold'); d.gap(0.002)
        for it in items:
            d.text('· ' + it, fs=8.1, dy=0.0143, indent=0.008)
        d.gap(0.006)

    d.text('4. Checking "multi-event transfer + same-event transfer" in the Proposal', fs=9.8, weight='bold'); d.gap(0.002)
    for it in [
        'Proposal.md does not contain the phrase "multi-event transfer + same-event transfer". After a word-by-word check, the closest items are the title and the three-level questions of §1, '
        'and the three experiments of §5.1 / §5.2 / §5.3: §5.1 scenario generalization within one hurricane (split by whole scenarios, split_A frozen), '
        '§5.2 cross-event generalization (whole hurricane held out, including a fixed-budget comparison), §5.3 limited-data transfer (adaptation with 1 / 3 / 5 target scenarios).',
        'Assessment: these three levels are worth keeping; the existing 486 simulations fully support them without any new data. '
        'The fixed-budget comparison of §5.2 (event diversity vs number of scenarios) and the few-shot adaptation of §5.3 are the two questions that most distinguish this project from the existing surrogate-model literature.',
        'Caveat (Proposal not changed; for your judgement only): the six events share the same terrain and grid, so "cross-event" means cross-forcing / cross-meteorology, '
        'not cross-catchment; cross-catchment transfer in the literature (Seleem 2023, do Lago 2023) is only partly analogous and the paper wording must distinguish them. '
        'Also, "1-2 complete hurricanes for final testing" in §5.2 is better phrased as 6-fold leave-one-out with only 6 events, to avoid a too-small test set.',
    ]:
        d.text('· ' + it, fs=8.1, dy=0.0143, indent=0.008)
    d.gap()

    d.text('5. Decision recommendations and items for you to choose', fs=9.8, weight='bold'); d.rule()
    d.text('Recommended direction: A (scenario -> 7-day main-period max depth / inundation extent) as the first task, immediately adding D (6-fold leave-one-out) and E (few-shot fine-tuning). '
           'B as the second stage; consider C only after the first two work.', fs=8.4, weight='bold')
    d.text('Recommended model: per-cell random forest / gradient boosting as first choice (few samples, interpretable, feature importance lets us discuss the roles of α β γ τ directly), '
           'U-Net as alternative and baseline; FNO left for stage B / C.', fs=8.4)
    d.text('Reasons: (1) the target maps already exist, no new heavy post-processing is needed; (2) 486 samples is a suitable size for tree models but small for large networks; '
           '(3) metrics for max depth and inundation extent (MAE, inundated-area error, CSI) are the easiest to interpret and match the metrics already set in Proposal §6; '
           '(4) once A is done, D and E cost almost nothing extra.', fs=8.4)
    d.gap()
    d.text('Items for you to choose / confirm:', fs=8.8, weight='bold')
    for i, q in enumerate([
        'Which main task: A / B / C / a combination? (A suggested)',
        'Preferred model: per-cell tree model, or go straight to U-Net? (tree model first suggested, U-Net as baseline)',
        'Cross-event evaluation: full 6-fold leave-one-out, or hold out only 1-2 events? (full 6 folds suggested)',
        'Should few-shot fine-tuning be done in this round, with 1 / 3 / 5 adaptation scenarios or another combination?',
        'Is "7-day main-period hourly-sampled max depth" acceptable as the Option A target (481 runs have no zsmax, so brief within-hour peaks may be missed; the 6 s041 runs use a different definition and are not mixed)?',
        'Keep the two inundation thresholds 0.1 and 0.3 m, or add more?',
        'Should I first produce a "dataset packaging plan" for your review before any modelling starts?',
    ], 1):
        d.text('%d) %s' % (i, q), fs=8.3, dy=0.0145, indent=0.008)

    # ===================== appendix =====================
    d.new('Appendix A: data fact check (all from verified metadata, not recomputed)')
    d.table(['Item', 'Value / notes'], [
        ['Simulations', '%d = 6 hurricanes x 81 scenarios; ledger / run_status / manifest all report %d ok; 0 log errors' % (n_run, S['ok'])],
        ['Per-run output', '%d hourly snapshots (t = 0...240 h, dt = 3600 s); first 73 (t <= 72 h) are spin-up, the 7-day main period is t = 73...240 h, 168 snapshots'
         % G['ampr']['n'][0]],
        ['Grid', '195 x 255, 200 m; active cells %s; water-level boundary cells %s; all six events share the same terrain and mask (single catchment, cannot test cross-catchment generalization)'
         % (G['n_active'][0], G['n_msk2'][0])],
        ['Field variables', 'h water depth (defined at all active cells, incl. thin dry-cell layer <= 0.05 m); zs water level (NaN at dry cells, a dry flag, not missing data); zb bed elevation has no NaN'],
        ['Max-depth target', 'event7d/hmax_event7d_hourlysampled.npy in each run directory = per-cell max over the 168 snapshots of the 7-day main period, ready to use; '
         '480 runs with dtmaxout = 0 have no zsmax, the 6 s041 runs use the old dtmaxout = 3600 definition; the two are not mixed'],
        ['Frames', '%s in total (incl. spin-up); %s main-period frames after removing spin-up. Note: adjacent hours within a scenario are strongly autocorrelated, '
         'and the 81 scenarios within an event differ by only 4 scalars -- they must not be treated as independent samples' % (format(frames_all, ','), format(frames_evt, ','))],
        ['Scenario factors', 'α tidal residual 0.8 / 1.0 / 1.2; β rainfall 0.7 / 1.0 / 1.3; γ low-frequency discharge 0.75 / 1.0 / 1.25; τ rainfall time shift -6 / 0 / +6 h; '
         's041 = unperturbed baseline; split_A 60 / 11 / 10 per event, frozen'],
        ['Forcing files', 'bzs 2401 x 12 (6-min boundary water level); dis 961 x 4 (15-min signed discharge, 868 rows for Irma); '
         'ampr 241 x 42 x 53 (1 km hourly rainfall mm/h)'],
        ['Station output', 'sfincs_his.nc: 1441 10-min time steps x 9 stations, of which 2 are outside the domain and NaN throughout (expected), 7 usable'],
        ['Inter-event differences', '; '.join('%s rain %.0f-%.0f mm / inundation %.0f-%.0f %%' % (e, rain[e][0], rain[e][1], wet[e][0], wet[e][1]) for e in EVS)],
        ['Generalization ceiling', 'only 6 real events -> at most 6-fold leave-one-out across events; the effective independent sample size is about 6, not 486, let alone %s frames' % format(frames_all, ',')],
        ['Needs additional data', '(1) validation against real inundation extent (high-water marks / satellite water) (2) more real hurricanes for denser cross-event folds '
         '(3) wind / pressure forcing (4) cross-catchment transfer (needs a second study area). None of these is done in this round; listed for your decision.'],
    ], colw=[0.13, 0.73], fs=7.2, lh=0.0125)

    d.gap()
    d.title = 'Appendix B: references for this survey'
    d.text('Appendix B: references for this survey (title / authors / year / DOI each verified; ★ = PDF stored in Model_paper/)', fs=9.4, weight='bold')
    d.gap(0.004)
    refs = [
        ('★ Leijnse, T., van Ormondt, M., Nederhoff, K., van Dongeren, A. (2021)',
         'Modeling compound flooding in coastal systems using a computationally efficient reduced-physics solver. Coastal Engineering 163, 103796. doi:10.1016/j.coastaleng.2020.103796',
         'The SFINCS model itself; Jacksonville / Irma test case.'),
        ('★ Lee, J., Sun, A. Y., Scanlon, B. R., et al. (2025)',
         'Probabilistic storm surge and flood-inundation modeling of the Texas gulf coast using SFINCS. Coastal Engineering 198, 104721. doi:10.1016/j.coastaleng.2025.104721',
         'Full text checked: pure SFINCS ensembles of 81 / 189 / 1000 members, no ML surrogate -- used only as a reference for the "ensemble idea".'),
        ('★ Zahura, F. T., Goodall, J. L., Sadler, J. M., et al. (2020)',
         'Training machine learning surrogate models from a high-fidelity physics-based model. Water Resources Research 56, e2019WR027038. doi:10.1029/2019WR027038',
         'Per-road-segment random forest surrogate of TUFLOW; 16 events for training / 4 for testing, split by whole events -- the direct template for Option A.'),
        ('Löwe, R., Böhm, J., Jensen, D. G., et al. (2021)',
         'U-FLOOD – Topographic deep learning for predicting urban pluvial flood water depth. Journal of Hydrology 603, 126898. doi:10.1016/j.jhydrol.2021.126898',
         '"Terrain + rainfall pattern -> max depth map" U-Net; explicitly evaluates rainfall events and locations unseen in training -- the alternative template for Option A.'),
        ('★ Kabir, S., Patidar, S., Xia, X., et al. (2020)',
         'A deep convolutional neural network model for rapid prediction of fluvial flood inundation. Journal of Hydrology 590, 125481. doi:10.1016/j.jhydrol.2020.125481',
         'CNN surrogate of LISFLOOD-FP predicting water depth, tested on two real floods; max-depth error 0-0.2 / 0-0.5 m for >99 % of cells.'),
        ('★ Bentivoglio, R., Isufi, E., Jonkman, S. N., Taormina, R. (2023)',
         'Rapid spatio-temporal flood modelling via hydraulics-based graph neural networks. HESS 27, 4227–4246. doi:10.5194/hess-27-4227-2023',
         'Autoregressive GNN, single-step prediction + curriculum lengthening of rollout steps; depth MAE 0.04 m on unseen terrain / unseen breach locations -- basis for the Option C training strategy.'),
        ('★ Sun, A. Y., Li, Z., Lee, W., et al. (2023)',
         'Rapid flood inundation forecast using Fourier neural operator. arXiv:2307.16090 (ICCV 2023 AI+HADR workshop). doi:10.48550/arXiv.2307.16090',
         'FNO surrogate of CREST-iMAP, 15-min step, inputs previous frames + rainfall + DEM, lead time up to 3 h; 6 events for training / 2 held out; FNO beats the U-Net baseline.'),
        ('★ Feng, D., Tan, Z., Lin, Z., et al. (2025)',
         'A comparative study of physics-informed and data-driven neural networks for compound flood simulation at river-ocean interfaces. JGR: Machine Learning and Computation 2, e2025JH000758. doi:10.1029/2025JH000758',
         'Compound flooding; Hurricane Irene, unseen in training, as an independent test: U-Net degrades clearly on the unseen extreme event, CNN-LSTM is the most robust -- the risk basis for Option D.'),
        ('★ Xu, Q., De Vos, L., Shi, F., Rüther, N., et al. (2025)',
         'Urban flood modeling and forecasting with deep neural operator and transfer learning. Journal of Hydrology 661, 133705. doi:10.1016/j.jhydrol.2025.133705',
         'Source-domain pretraining -> target-domain fine-tuning / domain adaptation; Berlin I data with 125 rainfall samples (100 / 13 / 12); error drops clearly once labelled target samples reach 4 or more -- basis for Option E.'),
        ('★ Seleem, O., Ayzel, G., Bronstert, A., Heistermann, M. (2023)',
         'Transferability of data-driven models to predict urban pluvial flood water depth in Berlin, Germany. NHESS 23, 809–822. doi:10.5194/nhess-23-809-2023',
         'Uses max depth from a 2D hydrodynamic model as truth, comparing CNN and random forest: RF is better within the training domain, CNN benefits more from transfer learning -- basis for the Option E model choice.'),
        ('Fraehr, N., Wang, Q. J., Wu, W., Nathan, R. (2022 / 2023 / 2024 / 2025)',
         'LSG series: doi:10.1029/2022WR032248; doi:10.1029/2022WR033836; Water Research 252, 121202, doi:10.1016/j.watres.2024.121202; '
         'J. Environmental Management 373, 123570, doi:10.1016/j.jenvman.2024.123570',
         'Spatio-temporal surrogate from low-fidelity hydrodynamics + EOF + sparse Gaussian process; the 2024 paper compares four mainstream surrogates and specifically evaluates events outside the training range; the 2025 paper discusses how to select training events.'),
        ('Eilander, D., Fraehr, N., Leijnse, T., de Goede, R. (2025)',
         'Surrogate flood models for compound flood risk assessments and early warning. EGU General Assembly 2025, abstract EGU25-5209.',
         'Conference abstract only (no full text): SFINCS + LSG surrogate for compound flooding, cases Brisbane and Charleston -- shows the "ML surrogate of SFINCS" direction is being pursued, but very few full texts are public.'),
        ('do Lago, C. A. F., Giacomoni, M. H., Bentivoglio, R., et al. (2023)',
         'Generalizing rapid flood predictions to unseen urban catchments with conditional generative adversarial networks. Journal of Hydrology 618, 129276. doi:10.1016/j.jhydrol.2023.129276',
         'cGAN with HEC-RAS as truth, trained on 10 catchments / tested on 5 unseen catchments -- reference for cross-catchment transfer (this project has a single catchment, so only partly analogous).'),
        ('Zhou, Q., Wu, W., Nathan, R., Wang, Q. J. (2022)',
         'Deep learning-based rapid flood inundation modeling for flat floodplains with complex flow paths. Water Resources Research 58, e2022WR033214. doi:10.1029/2022WR033214',
         'Spatio-temporal inundation modelling with 1D-CNN time series at representative points + U-Net spatial reconstruction; 1D-CNN beats LSTM -- a dimension-reduction alternative for Option B.'),
        ('★ Donnelly, J., Daneshkhah, A., Abolfathi, S. (2024)',
         'Physics-informed neural networks as surrogate models of hydrodynamic simulators. Science of the Total Environment 912, 168814. doi:10.1016/j.scitotenv.2023.168814',
         'PINN surrogates of LISFLOOD-FP and Delft3D; physics constraints outperform purely data-driven CNNs when data are limited.'),
        ('★ Rivera-Casillas, P., Dutta, S., Cai, S., et al. (2025/2026)',
         'A neural operator emulator for coastal and riverine shallow water dynamics. arXiv:2502.14782. doi:10.48550/arXiv.2502.14782',
         'Autoregressive neural operator (MITONet), inputs include the initial field, time-varying boundary conditions and bottom friction; long rollouts are feasible but depend on known boundary forcing -- feasibility and prerequisite for Option C.'),
        ('★ Radfar, S., Maghsoodifar, F., Moftakhari, H., Moradkhani, H. (2025)',
         'Integrating Newton\'s laws with deep learning for enhanced physics-informed compound flood modelling. arXiv:2507.15021. doi:10.48550/arXiv.2507.15021',
         'Preprint (no formal publication found): physics-constrained convolutional ConvLSTM for compound flooding; 6 historical storms = 4 train / 1 validation / 1 held-out test -- the same "few events" situation as this project.'),
    ]
    for a, b, c in refs:
        d.text(a, fs=7.4, dy=0.0122, weight='bold')
        d.text(b, fs=7.1, dy=0.0118, indent=0.008)
        d.text('→ ' + c, fs=7.1, dy=0.0118, indent=0.008, color='#333')
        d.gap(0.0025)
    d.gap(0.004)
    d.text('Items without full text are marked: Eilander et al. 2025 is only an EGU conference abstract; Radfar et al. 2025 is an arXiv preprint. '
           'Elsevier / Wiley copyright restrictions prevent legally downloading the publisher PDFs of Löwe 2021, do Lago 2023, Zhou 2022 and the Fraehr series; '
           'their DOIs and official links are kept in Model_paper/PAPERS.csv.', fs=7.2, dy=0.0122)

    d.close()

print('Wrote', OUT)
