# Example data

These real signal excerpts illustrate the EDF and label formats accepted by
DairySleepNet. They were selected from the 100 Hz, 30-second epoch arrays produced
by the research preprocessing pipeline and exported to new EDF files.

## Files and dimensions

The `aligned/` directory contains seven Cow IDs, `Cow1` through `Cow7`. Each ID
has three files:

| File pattern | Contents | Shape when loaded |
| --- | --- | --- |
| `CowN.edf` | Nine PSG channels | 9 channels × 27,000 samples |
| `CowN_ACC.edf` | Pressure and triaxial acceleration | 4 channels × 27,000 samples |
| `CowN.txt` | One numeric label per 30-second epoch | 9 labels |

All signal channels are sampled at 100 Hz. Each EDF contains nine epochs,
equivalent to 4.5 minutes of concatenated excerpts. The collection contains 14 EDF
files and seven label files, totaling approximately 5 MB and 63 epochs.

PSG channel order: `Abdomen`, `F3`, `C3`, `F4`, `C4`, `E1`, `E2`, `ChinL`, `ChinR`.
RumiWatch channel order: `Pressure`, `Move_X`, `Move_Y`, `Move_Z`.

## Selection and labels

For each source record, the first three valid Wake epochs, the first three valid
Sleep epochs and the first three valid Rumination epochs were selected. The nine
selected epochs were then ordered by their positions in the source record. PSG
and RumiWatch files use the same selected epochs and align with the label file.

The supplied text files use these input codes:

| Input code | Meaning | Prepared three-class label |
| --- | --- | --- |
| `0` | Wake | `0` |
| `1` | Sleep, with the study's sleep categories already merged | `1` |
| `4` | Rumination | `2` |

Use the preprocessing script's default `--label-format five` with these files.
The collection has 21 epochs per class. Each individual file has three epochs per
class; the label sequence reflects source order rather than a fixed class order.

## Interpretation

The selected epochs are nonconsecutive excerpts. Their concatenation does not
preserve the original gaps or a continuous behavioral timeline. The balanced
class counts also differ from the distribution of the full dataset.

The EDF headers contain placeholder patient/recording fields and the export date
`01.01.85`; this date is not the observation date. Cow filenames provide the
subject identifiers used by the example fold configuration.

See the main [README](../README.md#citation) for the associated paper and citation.
