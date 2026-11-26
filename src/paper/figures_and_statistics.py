"""
Rigenera le figure e le statistiche del manoscritto dagli output gia' salvati in results/, senza
rieseguire il modello. Va lanciato dopo src/reproducibility/results_reproduction.py.

Uso (dalla radice del repo):  python3 src/paper/figures_and_statistics.py
"""
import os
import sys

# Come nelle altre pipeline si lavora da src/, con i percorsi relativi a ../results.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_SRC_DIR, 'paper'))

import make_figures
import manuscript_statistics


def main():
    os.chdir(_SRC_DIR)
    out_dir = '../results'

    # Distribuzione di riferimento del grafico dei punteggi: le predizioni sul test set ZINC,
    # prodotte da results_reproduction.py (assenti se e' stato lanciato con --skip-zinc).
    reference = f'{out_dir}/test_data/predictions_raw.csv'
    make_figures.generate_all(out_dir=out_dir,
                              reference_predictions_path=reference if os.path.exists(reference) else None)
    manuscript_statistics.compute_all(out_dir=out_dir)


if __name__ == '__main__':
    main()
