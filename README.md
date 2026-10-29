# LLM-Based Generation of Standard Operating Procedures for Logistics Processes

Code and data for the master's thesis *"Large Language Model-Based Generation of
Standard Operating Procedures for Logistics Processes – A Controlled Comparative
Study"* (Rasha Mahodudi, TUM School of Management, 2026).

Four LLMs generated one SOP for each of 12 frozen logistics scenarios (48 SOPs).
Three LLM judges scored every SOP on an eight-criterion rubric (144 evaluations,
1,152 criterion ratings). Twelve SOPs were also rated by two logistics practitioners.

## What is original and what was added later

The repository root is the original experiment folder. The scripts, prompts,
outputs, logs and results there are the files used in the experiment. Files
added after the experiment sit in their own folders, listed below:

| Folder | Status | Content |
|---|---|---|
| repository root | **original** | generation scripts, analysis scripts, figure scripts, prompts, rubric, all outputs and logs |
| `wilcoxon_integer_rerun/` | **added after the analysis** | repeats the signed-rank tests on integer sums (see below); the thesis reports these values |
| `human_validation/` | **added** (final-phase validation, Oct 2026) | anonymised ratings of 12 SOPs by two practitioners (rater names removed), the HV_xx → SOP_xxx key, and the analysis script |
| `generation_prompt_v1.0.txt` | copy | the frozen generation prompt as a separate file; identical to the text embedded in the four generation scripts |

## Experiment

**Generation** (`*_experiment.py`; outputs and logs in `*_OFFICIAL_OUTPUTS/`)

| Model | Model ID | Access | Settings | Run date (log) |
|---|---|---|---|---|
| Qwen3.5-27B | `qwen/qwen3.5-27b` | OpenRouter (Alibaba) | temperature 0, max_tokens 4000 | 21 Sep 2026 |
| Llama 4 Maverick | `meta-llama/llama-4-maverick` | OpenRouter (DeepInfra) | temperature 0, max_tokens 4000 | 17 Sep 2026 |
| GPT-5.6 Sol | `gpt-5.6-sol` | OpenAI Responses API | reasoning effort none, max_output_tokens 4000, temperature not set | 21 Sep 2026 |
| Claude Sonnet 5 | `anthropic/claude-sonnet-5` | OpenRouter (Amazon Bedrock) | temperature 0, max_tokens 4000 | 21 Sep 2026 |

**Evaluation**: `evaluation_judge_prompt.txt` (v1.0), `evaluation_rubric_8_criteria.txt`,
`evaluation_models.txt`. Each SOP was anonymised (`SOP_001`–`SOP_048`, seed 20260929;
mapping with SHA-256 hashes in `FINAL_EVALUATION_MAPPING.csv`). The exact text sent
to the judges is in `FINAL_EVALUATION_INPUTS/`. Raw judge outputs and logs are in
`FINAL_EVALUATION_RESULTS/{GPT,CLAUDE,GEMINI}_JUDGE/` (all judges run on 29 Sep 2026).
`EVALUATION_PILOT/` contains the two-SOP pipeline pilot run before the final evaluation.

## Analysis scripts (original, run in this order)

| Step | Script | Output in `FINAL_EVALUATION_RESULTS/ANALYSIS/` |
|---|---|---|
| 1 | `evaluation_pilot.py` (judge descriptives; historical file name) | `judge_descriptive_statistics.csv` |
| 2 | `inter-judge agreement.py`, `pairwise_kappa.py`, `validate_krippendorff_alpha.py` | `inter_judge_agreement.csv`, `pairwise_weighted_kappa.csv` |
| 3 | `model_comparison.py` | `consensus_sop_scores_48.csv`, `generator_descriptive_statistics.csv` |
| 4 | `statistical_model_comparison.py`, `validate_step4.py` (SciPy) | `statistical_generator_comparison.csv` |
| 5 | `variant_analysis.py` | `variant_A_vs_B_analysis.csv` |
| 6 | `criterion_analysis.py`, `criterion_posthoc.py` | `criterion_level_analysis.csv`, `criterion_posthoc_comparisons.csv` |
| 7 | `self_evaluation_bias.py`, `same_family_judge_analysis.py` | `self_evaluation_bias_analysis.csv`, `same_family_judge_analysis.csv` |
| – | `figure_5_*.py` | `ANALYSIS_FIGURES/`, `ANYSIS_FIGURES/` |

`statistical_model_comparison.py` uses a normal approximation; `validate_step4.py`
repeats the tests with SciPy 1.18.1, and the thesis reports the SciPy values.

### Correction: signed-rank tests on integer sums

`consensus_sop_scores_48.csv` stores consensus means rounded to six decimals. When
paired differences such as −1/12 and +1/12 are rounded, they are no longer exact
ties, which changes the signed-rank statistic. This affected the overall RQ3 test
(original output T = 10, p = 1.000; correct T = 9.5, p = .875) and a few criterion
post-hoc p-values. None of the conclusions changed. `wilcoxon_integer_rerun/`
repeats all signed-rank tests on integer sums of the three judge totals.
These are rank-equivalent to the means and contain no rounding. The thesis
reports the values from these scripts. The original scripts and outputs are
kept unchanged.

## Reproducing the results

```
pip install -r requirements.txt
python validate_step4.py                      # RQ2 (SciPy)
cd wilcoxon_integer_rerun
python rq2_generator_comparison.py
python rq3_variant_ab.py
python rq4_criteria.py
python same_family_diagnostic.py
cd ..
python human_validation/human_validation_analysis.py
```

The generation scripts call commercial APIs. They need `OPENROUTER_API_KEY`
or `OPENAI_API_KEY` set as environment variables, and no keys are included
here. Hosted models can change over time, so a re-run will not produce
identical outputs.

## Integrity checks

`MANIFEST.csv` lists the SHA-256 hash of every file exactly as stored here;
`.gitattributes` stops Git from converting line endings, so these hashes stay
valid. The SHA-256 hashes in `FINAL_EVALUATION_MAPPING.csv` were computed on
the SOP texts with Unix (LF) line endings, whereas the stored files use Windows
(CRLF) line endings. To check them, convert CRLF to LF before hashing; all 48
SOPs then match their mapping hash.

## Limitations

- One output per model and scenario; no repeated generations.
- Finish-reason metadata were not retained for all interfaces. No truncated output was observed.
- Human validation covers 12 of the 48 SOPs with two raters and is a final-phase check, not the primary evaluation.
- Scenario texts were written by the author from public logistics sources (see thesis Section 4.2); the source documents are not redistributed here.
