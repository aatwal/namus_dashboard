# NamUs Missing Persons Dashboard

An interactive Python dashboard for exploring NamUs missing persons CSV exports.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8050 in your browser.

## Usage

1. Export a CSV from https://namus.nij.ojp.gov (register free, run a search, click Export All)
2. Drop the CSV onto the upload area in the app, or replace the path in app.py
3. Use the filters to toggle between all cases and children only, filter by county, or search by name/city

## Features

- Metric cards: total cases, children count, % children, top county
- Age group bar chart (child buckets 0–5 / 6–12 / 13–17, or adult buckets)
- Sex and race/ethnicity doughnut charts
- Top 8 counties horizontal bar chart
- Cases by year timeline
- Searchable, sortable, filterable case table
