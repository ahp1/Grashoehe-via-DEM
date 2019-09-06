Diese Repo besteht aus drei Python 3 Programmen um mit Hilfe von DEMs die Grashöhe zu bestimmen.
Als Referenz wird ein erster Überflug genommen, bei welchem das Gras frisch gemäht oder noch nicht gewachsen ist.

1. In einem ersten Schritt wird der zweite Überflug in x und y verschoben. Dies geschieht mit einer MEssung in QGIS von welchem die Werte in den 
Python Code übernommen werden.

2. Jetzt wird die Abweichung in z bestimmt. MIt Hilfe von Strassen oder Geleisen die als Referenz gelten.

3. Im letzten Schritt wird die Differenz der beiden DEMs berechnet.
