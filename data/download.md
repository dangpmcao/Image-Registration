# Downloading the Datasets

All datasets live in `/dataset/` (outside the repo, so they are never committed). See `dataset/README.md` for the contents of each folder and its licence.

Total size: about 24 GB. Tools used: `curl` (or `aria2c` for faster, resumable downloads), `unzip`, `tar`, and Python.

## 1. Direct downloads (no account needed)

Download each file into its folder, then extract it. In Git Bash, use `/d/...` paths with `tar`, because `tar` reads `D:` as a remote host.

| Folder | URL(s) |
|---|---|
| `mnist/` | `https://ossci-datasets.s3.amazonaws.com/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz` |
| `oasis_2d_neurite/` | `https://huggingface.co/datasets/DalcaLab/neurite-oasis/resolve/main/neurite-oasis.2d.v1.0.tar` |
| `acdc/` | `https://huggingface.co/datasets/YongchengYAO/ACDC/resolve/main/Images.zip`, `.../Masks.zip` |
| `sunnybrook/` | `https://downloads.sourceforge.net/project/cardiac-mr/Sunnybrook%20Cardiac%20MR%20Database%20{DICOM,Contours}Part{1,2,3}.zip` (6 files) |
| `learn2reg_lungct/` | `https://cloud.imi.uni-luebeck.de/s/o7LyCbJCie8fQ3B/download` (530 MB) |
| `learn2reg_oasis/` | `https://cloud.imi.uni-luebeck.de/s/xcZrLSQYtK68em8/download` (1.3 GB) |
| `learn2reg_abdomenctct/` | `https://cloud.imi.uni-luebeck.de/s/32WaSRaTnFk2JeT/download` (1 GB) |
| `learn2reg_nlst2023/` | `https://cloud.imi.uni-luebeck.de/s/pERQBNyEFNLY8gR/download` (11.7 GB) |
| `ixi/` | `http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar`, `.../IXI-T2.tar` |

Example:

```bash
cd /d/Optimization/dataset/learn2reg_oasis
curl -L -o OASIS.zip https://cloud.imi.uni-luebeck.de/s/xcZrLSQYtK68em8/download
unzip -q OASIS.zip && rm OASIS.zip
```

Always pass `-o <name>`. The Hugging Face and Learn2Reg links otherwise save files under a hash or under the name `download`.

Learn2Reg links come from the [Learn2Reg datasets page](https://learn2reg.grand-challenge.org/Datasets/). If a link stops working, take the new one from that page.

## 2. Breast DCE-MRI (TCIA Breast-MRI-NACT-Pilot) via NCI IDC

This step downloads only the baseline visit of each patient: the DCE series (t0/t1/t2) plus the segmentations, about 1.2 GB. The full collection is 19.5 GB. No account is needed.

```bash
pip install idc-index pandas
```

```python
from idc_index import IDCClient
c = IDCClient()
df = c.sql_query("""SELECT PatientID, StudyDate, SeriesInstanceUID, SeriesDescription, Modality, instanceCount
                    FROM index WHERE collection_id = 'breast_mri_nact_pilot'""")
d = df[df.StudyDate == df.groupby('PatientID').StudyDate.transform('min')]        # baseline visit
dce = d[(d.Modality == 'MR') & (d.instanceCount >= 150) & ~d.SeriesDescription.str.contains(':')]
seg = d[(d.Modality == 'SEG') & d.PatientID.isin(dce.PatientID)]
uids = list(dce.SeriesInstanceUID) + list(seg.SeriesInstanceUID)
c.download_from_selection(downloadDir='D:/Optimization/dataset/tcia_breast_nact',
                          seriesInstanceUID=uids,
                          dirTemplate='%PatientID/%Modality_%SeriesInstanceUID')
```

Each DCE series has 180 slices, which is 60 slices × 3 phases. Split the phases by the DICOM tag `AcquisitionTime`.

## 3. Manual downloads (form or login required)

| Dataset | How | Put in |
|---|---|---|
| **DIR-Lab 4DCT** (required for de Vos §8) | Fill in the request form on the [DIR-Lab page](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html). They send a Dropbox password. | `dirlab_4dct/` |
| FIRE (optional) | The official page is down. Use [hyper.ai](https://hyper.ai/en/datasets/5624) or [Dataset Ninja](https://datasetninja.com/fire-fundus). | `fire/` |
| ANHIR (optional) | Needs a grand-challenge account: [anhir.grand-challenge.org](https://anhir.grand-challenge.org/) | `anhir/` |
| ACDC full 4D (optional) | Needs a free account: [humanheart-project](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) | `acdc_full/` |
