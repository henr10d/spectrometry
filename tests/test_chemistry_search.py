import unittest
from src.chemistry import formula_mass, ELECTRON
from src.formulas import composition, dbe, enumerate_formulas, ppm_error
from src.fragmentation import enumerate_fragments, match_fragments


class ChemistrySearchTests(unittest.TestCase):
    def test_formula_search_contains_truth_and_respects_window(self):
        mass = formula_mass('C2H6O')
        formulas, nodes = enumerate_formulas(mass, {'C': 5, 'H': 20, 'O': 3, 'N': 2})
        self.assertIn('C2H6O', [x[0] for x in formulas])
        self.assertTrue(all(ppm_error(mass, formula_mass(x[0])) <= 10 for x in formulas))
        self.assertGreater(nodes, 0)
        self.assertEqual(dbe(composition('C6H6')), 4)
        with self.assertRaises(RuntimeError):
            enumerate_formulas(mass, {'C': 5, 'H': 20, 'O': 3}, max_nodes=1)
        with self.assertRaises(ValueError):
            composition('C6H12O6+')

    def test_fragments_conserve_parent_hydrogen_inventory(self):
        fragments = enumerate_fragments('CCO', max_cuts=1)
        by_atoms = {f['atoms']: f for f in fragments}
        self.assertEqual(by_atoms[(0,)]['formula'], 'CH3')
        self.assertEqual(by_atoms[(1, 2)]['formula'], 'CH3O')
        peak = formula_mass('CH3O') - ELECTRON
        matches = match_fragments([peak], fragments, hydrogen_shifts=(0,))
        self.assertTrue(any(fragments[x['fragment']]['formula'] == 'CH3O' for x in matches[0]))

    def test_ring_requires_two_cuts(self):
        self.assertEqual(enumerate_fragments('C1CCCCC1', max_cuts=1), [])
        self.assertGreater(len(enumerate_fragments('C1CCCCC1', max_cuts=2)), 0)
        with self.assertRaises(ValueError):
            enumerate_fragments('C1CCCCC1', max_cut_sets=1)


if __name__ == '__main__':
    unittest.main()
