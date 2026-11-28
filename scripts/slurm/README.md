# Dataset downloads through Slurm

Sources and breast-series selection follow `../../data/download.md`.
The cluster partition is named `gpu`, but these jobs request no GPUs.
Each task requests one CPU and 4 GiB RAM; the scheduler rounds CPU allocation
to a whole core (two hardware threads). At most three tasks run concurrently.
Jobs have a 72-hour limit and survive SSH disconnection and client shutdown.
Server outages are not automatically recovered; resubmit unfinished tasks.

| Array task | Dataset |
|---|---|
| 0 | Learn2Reg LungCT |
| 1 | Learn2Reg OASIS 3D |
| 2 | Learn2Reg AbdomenCTCT |
| 3 | Learn2Reg NLST2023 |
| 4 | IXI T1 |
| 5 | IXI T2 |
| 6 | Breast NACT baseline DCE and SEG |

Submit from the server containing the data:

```bash
cd /mnt/disk1/backup_user/dang.cpm/Optimization
mkdir -p dataset/.download_logs
sbatch --parsable --output="$PWD/dataset/.download_logs/%A_%a.log" \
    Image-Registration/scripts/slurm/download.sbatch "$PWD"
```

For selected failed tasks, add e.g. `--array=4-6%3` before the script path.
Do not resubmit while the original tasks are active. Per-dataset file locks
reject duplicate writers. Downloads resume existing archives and validate
ZIP CRCs or TAR structure before extracting. Archives are retained.
The official IXI host redirects HTTP to HTTPS but has an expired certificate
(checked 2026-11-28). Only IXI uses `curl --insecure`; TLS authenticity is not
verified for these public archives. Other sources retain TLS verification.
`.download_complete` records successful completion; completed tasks skip on rerun.
Breast downloads use a dedicated environment under `dataset/.download_idc_env`,
save the selected series and package versions, and check DICOM counts against
the IDC index. A completion marker does not replace a scientific dataset audit.

The initial submission on 2026-11-28 is array **3522**:

```bash
squeue -j 3522
sacct -j 3522 --format=JobID,State,ExitCode,Elapsed
tail -n 20 dataset/.download_logs/3522_*.log
```

DIR-Lab and the optional manual datasets are excluded; follow section 3 of
`data/download.md` for the required access steps.
