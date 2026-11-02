# Step 1 summary: Environment and repository validation

## Objective and implementation
Built a reproducible local environment and checked the AirLLM and Soup tools on a tiny random model.
- Python 3.12.10 in `.venv`, pinned in `.python-version`; 112 packages locked in `requirements.lock`.
- Seed used: 42. Scripts: `scripts/env_report.py`, `scripts/smoke_step01.py`, `scripts/check_api_keys.py`, `scripts/kumo_probe.py`.

## Observed environment
- Windows 11, 64 GB RAM, 12 threads, about 49 GB free on C: (95% full).
- GTX 1650 Max-Q, 4 GB VRAM, driver 573.22. PyTorch 2.11.0+cu128 with CUDA 12.8 is available.

## Repositories
| Tool | Commit | License file | Installed |
|---|---|---|---|
| AirLLM | b0b1a7efc033360e1d9e799f14287b8d4fe154bc | Apache-2.0 | airllm 4.0.0 (PyPI) |
| Soup | 433de4d7b1e1022113299ceb20e40ae66ef77eeb | Apache-2.0 | soup-cli 0.77.1 (PyPI) |

PyPI packages were installed; they were not verified as built from these exact commits.

## Results (tiny random model, interface checks only)
- Seeding: same seed repeated, different seed differed.
- PEFT LoRA: 5 steps on GPU, loss 10.44 to 10.29, peak VRAM 21.7 MB.
- AirLLM: generation worked. Training was not tested.
- Soup: 1-step LoRA run completed and the adapter reloaded in PEFT. Quantization was off.
- Soup telemetry: `cli.py` says it is opt-in through `SOUP_TELEMETRY=1`, though its help text describes an opt-out. Soup writes a local audit log to `~/.soup/audit.jsonl`.

## API keys (see `api_key_check.json`; each check also used a fake key)
- Working: HF_TOKEN, KAGGLE_KEY, TABPFN_TOKEN, TYPESAFE_API_KEY.
- Inconclusive: KUMO_API_KEY. The default `https://kumorfm.ai/api` now redirects to documentation, so real and fake keys fail identically.
- Kaggle: M5 download returns 403 until the competition rules are accepted.
- TabPFN: no license accepted yet for 2.5, 2.6, 3 or 3.5.
- KAGGLE_USERNAME is not validated: the KGAT token authenticates alone.

## Tests
28 of 28 passed; individual results are in `tests.log`.
Fixed during the step: `kaggle` failed to import because `six` was missing. The first test run missed this because `kaggle` was not in the import list; `six` was added and so was the test.

## Limitations and unresolved risks
- No real 3B to 8B model has been loaded, and QLoRA has not been run (`bitsandbytes` is not installed). Fitting on 4 GB is unproven and unlikely for 7B or larger.
- AirLLM speed on a real model was not measured.
- Timestral is not installed yet; its install method will be verified in Step 4.
- Kumo's API address is unresolved and blocks Step 6.
- During Kumo checks the key was sent to the default URL, whose redirect leads to docs.nvidia.com; HTTP clients may have forwarded the header.
- Soup is young and fast-changing, so its version must stay pinned.

## Acceptance criteria
- Versions recorded: met.
- Commits and licenses pinned and inspected: met for AirLLM and Soup.
- Imports, compatibility, minimal loading and seeding tests: met.
- Tool selection: partly met. Soup and PEFT are verified for LoRA on a tiny model only. AirLLM is verified for inference only.

## STOP: Awaiting user approval to proceed
