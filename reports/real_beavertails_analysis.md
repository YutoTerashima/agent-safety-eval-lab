# Real Dataset Analysis: BeaverTails Safety Rows

Source: [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)

This experiment uses 240 public BeaverTails rows from `30k_train`.
Raw prompts and responses are represented by hashes and structural features so the
repository can analyze safety coverage without redistributing sensitive generations.

- Unsafe rate in sample: 59.17%
- Average prompt length: 71.037 characters
- Average response length: 349.267 characters
- Top categories: violence,aiding_and_abetting,incitement (63), non_violent_unethical_behavior (40), financial_crime,property_crime,theft (24), discrimination,stereotype,injustice (21), privacy_violation (15)

Interpretation: the sample is useful for stress-testing rubric coverage because it
contains privacy, misinformation, unethical behavior, and harm-adjacent categories
that map naturally onto agent risk labels.
