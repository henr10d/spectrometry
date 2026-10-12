"""Index correctness can be tested independently of the native chemistry runtime."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.candidates import CandidateDB


class CandidateIndexTests(unittest.TestCase):
    def test_window_excludes_outside_mass_and_marks_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'candidates.parquet'
            pd.DataFrame({'mass':[100., 100.0009, 100.0011, 200.],
                          'key':['a','b','c','d'], 'smiles':['C','CC','CCC','CCCC']}).to_parquet(path)
            db = CandidateDB(path)
            ids, errors, fallback = db.retrieve([100.], ppm=10, minimum_da=0)
            self.assertEqual(db.key[ids].tolist(), ['a','b'])
            self.assertFalse(fallback)
            self.assertTrue(db.coverage([100.], 'b', ppm=10, minimum_da=0))
            self.assertFalse(db.coverage([100.], 'c', ppm=10, minimum_da=0))
            _, _, fallback = db.retrieve([150.], ppm=10, minimum_da=0)
            self.assertTrue(fallback)
            self.assertFalse(db.coverage([150.], 'a', ppm=10, minimum_da=0))
            with self.assertRaises(ValueError):
                db.retrieve([float('nan'), -1])


if __name__ == '__main__':
    unittest.main()
