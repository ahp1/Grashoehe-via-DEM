Diese Repo besteht aus drei Python 3 Programmen um mit Hilfe von DEMs die Grashöhe zu bestimmen.
Als Referenz wird ein erster Überflug genommen, bei welchem das Gras frisch gemäht oder noch nicht gewachsen ist.

1. In einem ersten Schritt wird der zweite Überflug in x und y verschoben. Dies geschieht mit einer Messung in QGIS von welchem die Werte in den 
Python Code übernommen werden.

2. Jetzt wird die Abweichung in z bestimmt. MIt Hilfe von Strassen oder Geleisen die als Referenz gelten.

3. Im letzten Schritt wird die Differenz der beiden DEMs berechnet.

4. Neu dazu kommt eine Berehnung für die kleine Schnittflächen und ein erster Ansatz einen Dichteschlüssel zu berechnen.

5. RGBVI, um ein Modell zwischen RPM und gemessener Höhe aus DEM Grashöhe zu erstellen.
