# Roadmap: Keyglide-Funktionen in Reflex

## Übernommene Funktionen

Stand 23.11.2026. [Entscheidungen zur Umsetzung](DECISIONS.md)

- Getrennte Messtexte für das echte Tempo (nie als Übung)
- Tage nach Ortszeit (Serie, Minuten heute, Tagespunkte)
- Anschläge werden schon während des Tippens gespeichert; abgebrochene Zeilen bleiben im Export, aber nicht im Training
- CSV- und JSON-Export mit Archiv, atomar, im Hintergrund, mit Fehlermeldung
- Heatmap mit Umschalter: Fehler / Tempo (gemessen) / Vorhersage (Modell)
- Automatischer App-Test: `./build.sh --smoke` bzw. `tools/smoke.sh`
- „Bewegung reduzieren“, VoiceOver-Beschriftungen, Hinweis nennt Zusatztaste und Finger

## Nacharbeiten aus der Prüfung (erledigt 23.11.2026)

- [x] **Offene Anschläge pro Zeile oder Instanz zuordnen.** *Umgesetzt als Sperre: Eine zweite Instanz auf denselben Daten startet nicht (`reflex.lock`), sondern zeigt einen Hinweis.* Schließt eine von zwei Reflex-Instanzen eine Zeile ab, löscht sie auch die offenen Anschläge der anderen. Beim Öffnen einer zweiten Instanz können bereits erfasste Anschläge zudem doppelt im Export landen. **Abnahme.** Mit zwei Instanzen auf derselben Testdatenbank tippen, in einer eine Zeile abschließen und die andere mitten in der Zeile beenden. Nach dem Neustart enthält der Export jeden Anschlag genau einmal. Alternativ den zweiten Zugriff auf die Datenbank zuverlässig verhindern. [Speicherung](core/reflex-core/src/store/mod.rs)
- [x] **Layoutwechsel mitten in einer Zeile korrekt erfassen.** *Die angefangene Zeile wird verworfen: Ihre Anschläge bleiben mit dem alten Layout als „abandoned“ im Export und zählen in keiner Heatmap. Danach kommen neue Zeilen für das neue Layout.* Beim Wechsel von DE ISO auf US ANSI werden aktuell auch die vor dem Wechsel getippten Anschläge dem neuen Layout zugeordnet. **Abnahme.** Nach einem Wechsel mitten in der Zeile zeigt der Export für jeden Anschlag das beim Tippen aktive Layout. Beide Heatmaps zählen nur die jeweils zugehörigen Versuche. [Zeilenabschluss](core/reflex-core/src/session.rs)
- [x] **Laufende Zeile exportieren.** *Spalte `status` (finished / open / abandoned) ersetzt `abandoned`.* CSV und JSON lassen die bereits gespeicherten Anschläge der aktuellen Zeile bis zum Abschluss oder Neustart aus. **Abnahme.** Mitten in der Zeile exportieren und einen Fehlversuch samt Korrektur in beiden Formaten als laufend erkennen. Nach Zeilenabschluss oder Neustart erscheint jeder Anschlag weiterhin genau einmal mit dem passenden Status. [Export](core/reflex-core/src/store/mod.rs)
- [x] **Historische Sitzungen dem richtigen lokalen Tag zuordnen.** *Jeder Batch speichert seinen UTC-Offset (Schema v3). Ältere Batches ohne Offset nutzen weiter den aktuellen.* Die Tageswerte und die Serie verwenden den aktuellen UTC-Offset auch für ältere Sitzungen. Dadurch kann eine Berliner Sitzung vom 11. Juli um 00:30 Uhr nach der Umstellung auf Winterzeit auf den 10. Juli rutschen. **Abnahme.** Eine Sitzung kurz nach Mitternacht behält über einen Sommerzeitwechsel hinweg ihren lokalen Kalendertag in Diagramm und Serie. [Tagesstatistik](core/reflex-core/src/ffi.rs) · [Kalendertage](core/reflex-core/src/stats.rs)

Für das Trainingsmodell bleibt die Schätzung des Lerngewinns laut Benchmark der nächste Ansatzpunkt. [Benchmark-Ergebnisse](DECISIONS.md)
