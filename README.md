**Autor(en):** Jorden Malecha (2922538, inf24040@lehre.dhbw-stuttgart.de)
**Repository:** https://github.com/JordenMalecha/XAI_Whisper_Abgabe
**Datensatz:** [gilkeyio/librispeech-alignments](https://huggingface.co/datasets/gilkeyio/librispeech-alignments) (LibriSpeech test-clean + MFA-Alignments; Original: [OpenSLR 12](https://www.openslr.org/12)). Wird vom Notebook automatisch gestreamt, kein manueller Download nötig.

# Wo hört Whisper hin?
### Zeitliche und spektrale Lokalisierung der Evidenz für einzelne Wörter in einem Encoder-Decoder-ASR-Modell

**Typ:** Post-hoc-Analyse · **Modelle:** `openai/whisper-tiny`, `openai/whisper-base` · **Daten:** LibriSpeech `test-clean` mit Wortgrenzen aus dem Montreal Forced Aligner (MFA) · **Code:** [`whisper_xai.ipynb`](whisper_xai.ipynb)

## 1 Fragestellung

> Welche Zeitabschnitte und akustischen Merkmale sind für die Transkription einzelner Wörter durch Whisper entscheidend, und wie gut lokalisieren Input-Attribution und Attention diese Evidenz im Vergleich zu den tatsächlichen Wortgrenzen?

Die meisten XAI-Auswertungen haben keine objektive Referenz. Hier gibt es eine: Forced Alignment liefert für jedes Wort ein Zeitsegment $[s, e)$. Damit lässt sich messen, ob eine Erklärung „an der richtigen Stelle“ liegt.

| | Hypothese | Prüfung |
|---|---|---|
| **H1** | Die Attribution für ein Wort liegt deutlich über Zufallsniveau im Wortsegment, plus wenige 100 ms Kontext. | `mass_in` vs. `chance` (Wilcoxon), Pointing Game, ereigniskorreliertes Profil |
| **H2** | Nur wenige Cross-Attention-Köpfe des Decoders lokalisieren gut, und es sind die von OpenAI für Zeitstempel genutzten `alignment_heads`. | Pointing-Accuracy pro Kopf |
| **H3** | Frühe Encoder-Schichten arbeiten lokal und diagonal, späte global. | Lokalität und Aufmerksamkeitsdistanz pro Schicht (Spearman) |
| **H4** | Attention ist schwächer kausal mit der Ausgabe verknüpft als Attribution. | Deletion-Kurven, Kopf-Ablation |

## 2 Modell und Erklärungsziel

Whisper (Radford et al., 2023) wandelt Audio in ein 80-Band-Log-Mel-Spektrogramm um: 10 ms pro Frame, **immer auf 30 s aufgefüllt**, also 3000 Frames. Zwei Conv-Schichten halbieren die Zeitauflösung, der Encoder arbeitet dann mit 1500 Frames à 20 ms. Ein autoregressiver Decoder liest den Encoder über Cross-Attention.

* **Ziel der Erklärung:** $f(X)=\log p(\text{Wort}\mid X)=\sum_{t\in\text{Wort}}\log p(y_t\mid y_{<t},X)$. Gerechnet wird mit Teacher Forcing auf Whispers eigene Transkription. Ausgewertet werden nur Wörter, die Whisper korrekt erkannt hat, denn nur dort ist die MFA-Grenze die richtige Referenz. BPE-Tokens werden zu Wörtern zusammengefasst.
* **Baseline $X_0$ = Stille:** das Padding-Spektrum der jeweiligen Äußerung. Der Feature-Extractor setzt Werte unter (Maximum − 8) auf diesen Mindestwert, aufgefüllte Nullen landen also genau dort. $X_0$ ist damit für das Modell bekannte Stille und keine künstliche Null-Eingabe.

## 3 Methoden

| Methode | Was sie berechnet | Warum | Schwäche |
|---|---|---|---|
| **Integrated Gradients** (Sundararajan et al., 2017) | $(x-x_0)\odot\int_0^1\nabla f(x_0+\alpha(x-x_0))\,d\alpha$, 256 Stützstellen | Erfüllt die Axiome *Completeness* und *Sensitivity*. Die Completeness wird im Notebook pro Wort geprüft, die Konvergenz zusätzlich für 16–256 Stützstellen. | Hängt von der Baseline ab, teuer |
| **Gradient × Δ** (Shrikumar et al., 2016) | $\nabla f(x)\odot(x-x_0)$ | Billige lokale Näherung, Vergleich zu IG | Wird bei Sättigung ungenau |
| **Zeit-Occlusion** (Zeiler & Fergus, 2014) | Abfall von $\log p$, wenn ein 100-ms-Fenster durch $X_0$ ersetzt wird | Modellagnostisch und kausal | Grobe Auflösung, erzeugt Eingaben außerhalb der Trainingsverteilung |
| **Band-Occlusion** | Wie oben, aber 8 Mel-Bandgruppen nur im Wortsegment | Kausale Gegenprobe zur spektralen IG-Analyse | Wie oben |
| **Cross-Attention** | Attention der Decoder-Queries eines Wortes über die Encoder-Frames, pro Schicht und Kopf | Whisper erzeugt seine Wort-Zeitstempel aus genau diesem Signal (alignment heads + DTW) | Zeigt Informationsfluss, keine Wichtigkeit (Jain & Wallace, 2019) |
| **Analyse der Encoder-Köpfe** (Clark et al., 2019; Voita et al., 2019) | Pro Kopf: Lokalität (±40 ms), Vorgänger/Nachfolger, mittlere Distanz, Entropie, Masse auf dem Padding, Bindung an das eigene Wort (`within_lift`), Masse auf Pausen | Beschreibt, welche Art von Lokalisierung im Encoder stattfindet | Nur beschreibend |
| **Kopf-Ablation** (Michel et al., 2019) | Ausgabe eines Kopfes vor `out_proj` auf 0 setzen → Δ log p, ΔWER | Kausaler Test für H2 und H4 | Null-Ablation ist selbst ein Eingriff außerhalb der Verteilung |

**Metriken:** Ein Relevanzprofil $p(t)$ wird auf Summe 1 normiert, für Attribution als |·| über alle Mel-Bänder summiert. Daraus berechnen wir:

* `mass_in`: Masse im Wortsegment,
* `lift`: `mass_in` geteilt durch den Zufallswert (Wortdauer / Äußerungsdauer),
* Pointing Game: Liegt das Maximum im Wort (±20 ms)?,
* Masse in den 300 ms vor bzw. nach dem Wort,
* Abstand des Peaks zur Wortmitte.

Das **ereigniskorrelierte Profil** mittelt die Relevanz über alle Wörter auf einer relativen Zeitachse (0 = Wortanfang, 1 = Wortende). Der **Deletion-Test** (Samek et al., 2017) entfernt die laut Methode relevantesten Frames, und zwar 0,25- bis 4-mal so viele wie das Wort lang ist. Als Schranken dienen *Zufall* und *Oracle*, wobei Oracle die Frames nach Nähe zur Wortmitte entfernt. Gleich bewertete Frames werden zufällig gereiht.

**Statistik:** Wörter derselben Äußerung sind nicht unabhängig. Tests laufen deshalb auf Äußerungsebene (Mittel je Äußerung, Wilcoxon), Konfidenzintervalle stammen aus einem Cluster-Bootstrap über Äußerungen, und mehrere gleichartige Tests werden nach Holm korrigiert. Referenz und Hypothese werden vor WER und Wortabgleich mit Whispers `EnglishTextNormalizer` normalisiert.

## 4 Ausführung

Getestet unter Windows mit einer GTX 1660 SUPER (6 GB VRAM), Python 3.14 und CUDA 12.8:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` pinnt alle Versionen, mit denen die Ergebnisse erzeugt wurden (PyTorch 2.11.0+cu128, Transformers 5.23.0, Datasets 5.2.0 u. a.). Danach im Notebook den Kernel der `.venv` wählen und von oben ausführen.

* **Vollständiger Lauf:** 100 Äußerungen ≤ 15 s für die Lokalisierung, davon 25 für Deletion und Ablation und 3 für den IG-Konvergenzcheck. Rechenzeit auf der GTX 1660 SUPER: 18,7 Stunden, überwiegend für Integrated Gradients mit 256 Stützstellen (ca. 150 s pro Äußerung bei tiny, ca. 320 s bei base).
* **Ergebnis-Cache:** Der Lauf speichert alle Ergebnisse zusammen mit der Konfiguration in `results/cache_res.pkl`. Mit `LOAD_CACHED = True` (Standard) lädt das Notebook diesen Cache, sofern `CFG` identisch ist; alle Auswertungen laufen dann in Sekunden. Bei geänderter `CFG` wird neu gerechnet. Der Cache ist wegen seiner Größe nicht im Repository (`.gitignore`).
* **Zwischenstand:** Nach jedem Modell wird ein Zwischenstand gespeichert (`results/cache_part_*.pkl`). Bricht der Lauf bei base ab, wird tiny beim nächsten Start nicht neu gerechnet.
* **Feste Revisionen:** Datensatz und Modelle werden in einer festen Hugging-Face-Revision geladen (`DATASET`, `MODEL_REVISIONS` in der Setup-Zelle).
* **Probelauf:** Umgebungsvariable `WHISPER_XAI_QUICK=1` setzen (10 Äußerungen, 3 für Deletion/Ablation, weniger IG-Schritte). Die Ausgaben landen in `results_quick/` und überschreiben die Hauptergebnisse nicht.
* **Speicher:** `batch` und `ig_batch` in `CFG` sind für 6 GB gewählt. Bei Speicherfehlern beide auf 1 setzen; die Ergebnisse ändern sich dadurch nicht.
* Alle Parameter stehen in `CFG`.
* Ausgaben in `results/`: Abbildungen, `model_comparison.csv`, `summary.json` und je Modell die Zwischenergebnisse als CSV/JSON: verwendete Äußerungen (`*_utterances.csv`), Wortmetriken, Deletion, Ablation samt generierter Texte, Kennzahlen je Cross-Attention- und Encoder-Kopf, Band-Occlusion sowie `ig_convergence.csv`. Damit ist die Auswertung ohne neuen Lauf nachvollziehbar.

## 5 Ergebnisse

Vollständiger Lauf mit 100 Äußerungen (1756 bzw. 1779 korrekt erkannte Wörter), davon 25 für Deletion und Ablation. Alle Werte stammen aus `results/model_comparison.csv` und den Notebook-Abschnitten 7c–13.

| | tiny | base |
|---|---|---|
| WER (Whisper-normalisiert) | 0,075 | 0,052 |
| IG-Completeness-Fehler (Median, 256 Stützstellen) | 0,2 % | 0,3 % |
| IG `lift` (Median) / Pointing (Zufall) | 6,3 / 0,79 (0,05) | 5,4 / 0,77 (0,05) |
| Occlusion Pointing | 0,92 | 0,86 |
| Bester Cross-Kopf / Mittel der alignment heads / Mittel aller Köpfe (Pointing) | 0,99 / 0,58 / 0,31 | 0,99 / 0,73 / 0,22 |
| Encoder `local`: erste → letzte Schicht | 0,41 → 0,16 | 0,35 → 0,12 |
| Deletion-AUC: IG / Occlusion / CrossAttn (alignment heads) / Oracle / Zufall | −3,67 / −3,88 / −3,42 / −4,31 / −0,36 | −2,86 / −2,98 / −3,04 / −3,83 / −0,23 |
| ΔWER (gekappt): alignment heads / Zufallsmengen (Median, Spanne) | +0,74 / +0,22 (0,04–0,74) | +0,52 / +0,05 (0,01–0,41) |

* **H1 – bestätigt.** `mass_in` liegt in allen 100 Äußerungen über dem Zufallsniveau (Holm-korrigiert p < 1e-16), mit symmetrischem Kontext von je ca. 0,09–0,11 Masse in den 300 ms vor und nach dem Wort. Occlusion lokalisiert am schärfsten. Funktionswörter werden bei gleicher Wortdauer etwas diffuser lokalisiert (`lift` ca. 15–19 % niedriger), aber gleich gut getroffen.
* **H2 – teilweise bestätigt.** Nur 7/24 bzw. 9/48 Cross-Attention-Köpfe lokalisieren gut, fast alle in späten Schichten. Die `alignment_heads` sind im Mittel deutlich besser als der Durchschnitt, aber nicht die besten Köpfe; in base ist einer ein Padding-Sink.
* **H3 – teilweise bestätigt (explorativ).** Der geplante Test über die Köpfe ist nach globaler Korrektur in keinem Modell signifikant. Ein nach dem ersten Lauf ergänzter Test zeigt, dass die Lokalität bei jeder der 100 Äußerungen mit der Schichttiefe sinkt; das ist ein konsistentes Muster über Eingaben, nicht über Köpfe. Die Aufmerksamkeitsdistanz wächst nur bei base.
* **H4a – teilweise bestätigt (explorativ getestet).** Über alle Köpfe gemittelte Attention ist im Deletion-Test signifikant weniger treu als IG. Die Alignment-Köpfe sind bei base treuer als IG und gleichauf mit Occlusion; bei tiny liegen sie hinter IG, nach globaler Korrektur aber nicht signifikant.
* **H4b – bestätigt.** Die kausal wichtigsten Einzelköpfe lokalisieren schlecht (Pointing ≈ 0,1), gut lokalisierende Köpfe haben einzeln kaum Wirkung (Spearman ρ = −0,56 / −0,54, nach globaler Korrektur nur bei base signifikant). Als Gruppe sind die Alignment-Köpfe dagegen wichtig: Gemeinsam abgeschaltet senken sie log p um 1,56; bei base schadet keine Zufallsmenge gleicher Größe so stark.
* **Spektral** zählt vor allem 410–745 Hz (u. a. Bereich des ersten Formanten), auch pro Einheit entfernter Energie. Die Korrelation von |IG| mit der Energie (0,67) ist größtenteils ein Artefakt der IG-Definition; der reine Gradient korreliert kaum (0,13–0,15).

**Statistik:** Tests, die erst nach dem ersten Lauf ergänzt wurden (H3 auf Äußerungsebene, gepaarte Deletion-Vergleiche, Regression zur Wortdauer), sind als explorativ gekennzeichnet. Notebook-Abschnitt 13b korrigiert alle 40 Tests gemeinsam nach Holm: 5 von 33 nominal signifikanten Tests verlieren dabei die Signifikanz.

## 6 Grenzen und Reflexion

* **Was als Referenz gilt:** MFA-Grenzen sind selbst modellbasiert und auf etwa 10–20 ms genau. Ein Profil, das über die Grenze hinausreicht, ist nicht automatisch falsch, denn wegen Koartikulation steckt akustische Information über ein Wort auch in den Nachbarsegmenten.
* **Abhängigkeit von der Baseline:** IG und Occlusion beantworten die Frage „Relevanz *relativ zu Stille*“. Eine andere Baseline, etwa Rauschen, kann andere Karten liefern.
* **Out-of-Distribution-Problem** (Hooker et al., 2019): Das Entfernen von Frames erzeugt Eingaben, die das Modell im Training nie gesehen hat. Ein Abfall von $\log p$ kann daher teilweise auf diese Verteilungsverschiebung zurückgehen und nicht auf fehlende Information. Die Stille-Baseline mildert das, beseitigt es aber nicht.
* **Selektionsbias:** Weil nur korrekt erkannte Wörter ausgewertet werden, fehlen die schwierigen Fälle. Funktionswörter sind zudem stark vom Kontext des Decoder-Sprachmodells bestimmt. Eine schlechte akustische Lokalisierung ist dort plausibel und kein Fehler der Methode.
* **Attention ist nicht gleich Erklärung:** Attention auf dem 30-s-Padding (*Attention Sink*) oder hohe Lokalisierung ohne Wirkung bei der Ablation zeigen genau diese Grenze. H4 macht sie messbar.
* **Skalierung:** Auch `whisper-base` ist klein. Ob die Befunde auf `large` übertragbar sind, bleibt offen.
* **Stichprobe und Rechenaufwand:** Deletion und Ablation beruhen auf 25 Äußerungen, die Gruppenablation auf fünf Zufallsmengen und einem Seed. Die Bootstrap-Intervalle erfassen nur die Unsicherheit über Äußerungen. Wiederholungen mit anderen Seeds waren bei 18,7 Stunden Laufzeit nicht möglich. Mit 25 Clustern sind Perzentil-Intervalle eher zu schmal.
* **Explorative Analysen:** Einige Tests wurden nach dem ersten Lauf ergänzt. Ihre Ergebnisse müssten an neuen Äußerungen bestätigt werden.
* **Vorzeichen und Auflösung:** Für die Zeitprofile werden Beträge |IG| und |Grad×Δ| verwendet, bei Occlusion nur der positive Teil; Evidenz gegen ein Wort zählt bei IG also als Relevanz. Attention arbeitet mit 20 ms, Attribution mit 10 ms. Die Methoden sind deshalb nicht vollständig vergleichbar.
* **Ablation:** Köpfe werden auf 0 gesetzt (Null-Ablation, selbst ein Eingriff außerhalb der Trainingsverteilung), und die Zufallsmengen stammen aus allen Schichten statt aus denselben wie die Alignment-Köpfe.
* **Kein Sanity-Check:** Ein Modell-Randomisierungstest (Adebayo et al., 2018) wurde nicht durchgeführt.
* **Zeitstempel:** Whispers eigenes Verfahren (Medianfilter und DTW) wurde nicht nachgebildet; Aussagen beziehen sich nur auf die rohe Attention.

**Offene Punkte, die einen neuen Lauf erfordern:** Mittelwert-Ablation und schichtgleiche Zufallsmengen, eine zweite Baseline für den Deletion-Test, mehr Äußerungen und Seeds für Deletion und Ablation, Zeitprofile auf gemeinsamer Auflösung, Modell-Randomisierungstest, Auswertung falsch erkannter Wörter sowie größere Modelle und andere Daten.

## Quellen

* Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML. arXiv:2212.04356
* Panayotov, V. et al. (2015). *LibriSpeech: An ASR corpus based on public domain audio books.* ICASSP.
* McAuliffe, M. et al. (2017). *Montreal Forced Aligner.* Interspeech. Alignments: Lugosch, L. et al. (2019), Interspeech; HF-Datensatz `gilkeyio/librispeech-alignments`
* Sundararajan, M., Taly, A., Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* ICML. arXiv:1703.01365
* Shrikumar, A. et al. (2016). *Not Just a Black Box: Learning Important Features Through Propagating Activation Differences.* arXiv:1605.01713
* Zeiler, M., Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks.* ECCV. arXiv:1311.2901
* Samek, W. et al. (2017). *Evaluating the Visualization of What a Deep Neural Network Has Learned.* IEEE TNNLS. arXiv:1509.06321
* Adebayo, J. et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS. arXiv:1810.03292
* Hooker, S. et al. (2019). *A Benchmark for Interpretability Methods in Deep Neural Networks (ROAR).* NeurIPS. arXiv:1806.10758
* Becker, S. et al. (2018/2024). *Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals (AudioMNIST).* arXiv:1807.03418
* Clark, K. et al. (2019). *What Does BERT Look At? An Analysis of BERT's Attention.* BlackboxNLP. arXiv:1906.04341
* Voita, E. et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL. arXiv:1905.09418
* Michel, P., Levy, O., Neubig, G. (2019). *Are Sixteen Heads Really Better than One?* NeurIPS. arXiv:1905.10650
* Jain, S., Wallace, B. (2019). *Attention is not Explanation.* NAACL. arXiv:1902.10186
* Wiegreffe, S., Pinter, Y. (2019). *Attention is not not Explanation.* EMNLP. arXiv:1908.04626
