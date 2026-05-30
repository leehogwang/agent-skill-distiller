import unittest

from split_train_test import split_foods_by_distance


class SplitFoodsByDistanceTests(unittest.TestCase):
    def test_pairs_random_train_food_with_farthest_remaining_test_food(self):
        foods = ["rice", "soup", "cake", "sashimi"]
        embeddings = {
            "rice": [1.0, 0.0],
            "soup": [0.9, 0.1],
            "cake": [-1.0, 0.0],
            "sashimi": [-0.9, -0.1],
        }

        train, test = split_foods_by_distance(foods, embeddings, seed=0)

        self.assertEqual(train, ["sashimi", "cake"])
        self.assertEqual(test, ["soup", "rice"])

    def test_odd_count_keeps_extra_food_in_train(self):
        foods = ["a", "b", "c"]
        embeddings = {
            "a": [1.0, 0.0],
            "b": [-1.0, 0.0],
            "c": [0.0, 1.0],
        }

        train, test = split_foods_by_distance(foods, embeddings, seed=1)

        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 1)
        self.assertEqual(sorted(train + test), sorted(foods))


if __name__ == "__main__":
    unittest.main()
