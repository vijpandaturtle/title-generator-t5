import urllib
import arxiv
import csv
import pandas as pd
import datetime
import uuid

random_uuid = uuid.uuid4()

search = arxiv.Search(
  query = "artificial intelligence",
  max_results = 20000,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

# Collect results into a list
results_list = []
for result in search.results():
    results_list.append({
        'title': result.title,
        'summary': result.summary
})

# Create a DataFrame from the results
df = pd.DataFrame(results_list)
df.to_csv(f'./data/title_generator_dataset_v{}.csv'.format(random_uuid))