"""Download the baseline DCE/SEG selection specified in data/download.md."""
import sys
from pathlib import Path

import pandas as pd
from idc_index import IDCClient

root = Path(sys.argv[1])
client = IDCClient()
df = client.sql_query("""
    SELECT PatientID, StudyDate, SeriesInstanceUID,
           SeriesDescription, Modality, instanceCount
    FROM index WHERE collection_id = 'breast_mri_nact_pilot'
""")
baseline = df[df.StudyDate == df.groupby("PatientID").StudyDate.transform("min")]
dce = baseline[
    (baseline.Modality == "MR")
    & (baseline.instanceCount >= 150)
    & ~baseline.SeriesDescription.str.contains(":", na=False)
]
seg = baseline[
    (baseline.Modality == "SEG") & baseline.PatientID.isin(dce.PatientID)
]
selection = pd.concat([dce, seg]).drop_duplicates("SeriesInstanceUID")
if selection.empty:
    raise RuntimeError("No baseline DCE/SEG series selected")
selection.to_csv(root / ".download_selection.csv", index=False)
print(f"Selected {dce.PatientID.nunique()} patients, {len(selection)} series", flush=True)
client.download_from_selection(
    downloadDir=str(root),
    seriesInstanceUID=selection.SeriesInstanceUID.tolist(),
    dirTemplate="%PatientID/%Modality_%SeriesInstanceUID",
)
for row in selection.itertuples():
    directory = root / row.PatientID / f"{row.Modality}_{row.SeriesInstanceUID}"
    count = len(list(directory.rglob("*.dcm")))
    if count != int(row.instanceCount):
        raise RuntimeError(f"Incomplete series {directory}: {count}/{row.instanceCount} DICOM files")
