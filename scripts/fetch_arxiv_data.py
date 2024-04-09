import urllib
import arxiv
import csv
import pandas as pd
import datetime
import random

num = random.random()

search = arxiv.Search(
  query = "artificial intelligence",
  max_results = 100,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

# Collect results into a list
results_list = []
for result in search.results():
    results_list.append({
        'titles': result.title,
        'abstracts': result.summary
})

# Create a DataFrame from the results
df = pd.DataFrame(results_list)
df.to_csv(f'../data/title_generator_dataset_v_{num}.csv')