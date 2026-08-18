# Test fixture databases

Four of the smallest SQLite databases from the [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev)
benchmark (CC BY-SA 4.0), committed here so CI can run the DeepEval execution-accuracy
regression gate without downloading the full ~3.3GB dataset on every push.

| Database | Size |
|---|---|
| `superhero` | 232 KB |
| `student_club` | 2.5 MB |
| `toxicology` | 2.6 MB |
| `thrombosis_prediction` | 7 MB |

The full dataset (500 questions across 11 databases) is fetched separately via
`scripts/fetch_bird_data.py` for the full eval run (`eval/run_bird_eval.py`) - not
needed for the CI regression gate itself.

Source: [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev),
distributed under CC BY-SA 4.0.
