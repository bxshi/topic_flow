Usage:

0. Clone project

```bash
git clone https://github.com/bxshi/topic_flow.git
```

1. Get some reddit discussions

```bash
cd topic_flow/data/reddit
python3 reddit_crawler.py
```

2. Run the example

```bash
cd ../../
python3 -m tofo.examples.BOWExample ./data/reddit/til_baby_otters_cant_swim_at_first_but_their.json
fdp BOWExample.dot -Tsvg -o BOWExample.svg
```

Then the `BOWExample.svg` is the generated graph file.
