"""Small synthetic checks for reference dataset serialization and frozen IDs."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_reference_benchmarks import hellaswag_rows, select_quality, write_jsonl


class ReferencePreparationTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_context_without_extra_logical_lines(self):
        rows = [{'context': 'first\u2028second\u0085third'}, {'context': 'another'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'records.jsonl'
            spec = write_jsonl(path, rows)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual([json.loads(line) for line in lines], rows)
            self.assertEqual(spec['examples'], 2)

    def test_hellaswag_duplicate_original_ind_is_not_dropped(self):
        row = {'ind': 180, 'ctx': 'A task then', 'ctx_a': 'A task', 'ctx_b': 'then',
               'activity_label': 'activity', 'source_id': 'group', 'split': 'val',
               'split_type': 'indomain', 'label': '2', 'endings': ['one', 'two', 'three', 'four']}
        converted = hellaswag_rows([row, {**row, 'ctx_b': 'next'}])
        self.assertEqual(len(converted), 2)
        self.assertEqual([r['ind'] for r in converted], [180, 180])
        self.assertEqual(len({r['benchmark_id'] for r in converted}), 2)
        self.assertEqual(converted[0]['choices'], row['endings'])
        self.assertEqual(converted[0]['gold_index'], 2)
        self.assertEqual(converted[0]['completion_char_lengths'], [3, 3, 5, 4])

    def test_quality_seed_sample_is_document_disjoint_and_order_stable(self):
        rows = [{'document_id': f'doc-{i:03d}', 'question_unique_id':f'q-{i:03d}-{j}',
                 'id':f'item-{i}-{j}'} for i in range(115) for j in range(2)]
        selected = select_quality(rows, seed=19)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({r['document_id'] for r in selected}), 100)
        self.assertEqual([r['id'] for r in selected],
                         [r['id'] for r in select_quality(list(reversed(rows)), seed=19)])


if __name__ == '__main__':
    unittest.main()
