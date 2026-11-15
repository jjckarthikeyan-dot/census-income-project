# Census Income Classification and Customer Segmentation

**Python · Machine Learning · Data Analysis**

A portfolio project using U.S. Census data to explore two questions: can income groups be classified from demographic and employment features, and what meaningful population segments can be identified without using the income label?

## What this project covers

- **Classification:** predict whether an observation falls above the $50K income threshold.
- **Segmentation:** group observations using their characteristics, without using the income target for clustering.
- **Evaluation:** examine class imbalance and sample weights, alongside ROC-AUC, PR-AUC, and balanced accuracy.
- **Reporting:** export model metrics, feature importance, and segment profiles for inspection.

This is an analytical demonstration, not a production decision-making system. Demographic predictions should not be used to determine eligibility for jobs, credit, housing, or other high-impact services.

## Repository layout

| Path | Purpose |
| --- | --- |
| \`src/train_classifier.py\` | Classification workflow |
| \`src/segmentation.py\` | Segmentation workflow |
| \`requirements.txt\` | Python dependencies |
| \`REPORT.md\` | Detailed methodology and interpretation |
| \`outputs/\` | Generated model and analysis artifacts |

## Quick start

\`\`\`bash
git clone https://github.com/jjckarthikeyan-dot/census-income-project.git
cd census-income-project
python -m venv .venv
\`\`\`

Activate the environment:

- macOS/Linux: \`source .venv/bin/activate\`
- Windows PowerShell: \`.venv\Scripts\Activate.ps1\`

Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Place the source dataset in the \`data/\` directory, including:

\`\`\`text
data/census-bureau.data
data/census-bureau.columns
\`\`\`

## Run the workflows

\`\`\`bash
python src/train_classifier.py
python src/segmentation.py
\`\`\`

The workflows are documented as producing:

| Artifact | Purpose |
| --- | --- |
| \`outputs/classification_metrics.json\` | Classification evaluation |
| \`outputs/feature_importance.csv\` | Model feature interpretation |
| \`outputs/income_classifier.joblib\` | Serialized classifier |
| \`outputs/segment_profile.csv\` | Segment characteristics |
| \`outputs/segmentation_summary.json\` | Segmentation summary |
| \`outputs/census_with_segments_sample.csv\` | Sample labeled observations |

Generated file availability depends on successful execution and the dataset provided. See [REPORT.md](REPORT.md) for further details.

## Engineering considerations

Sample weighting and imbalanced classes matter when interpreting Census-based predictions. Classification performance should be evaluated beyond raw accuracy. The clustering workflow intentionally separates descriptive segmentation from supervised income prediction.

## Project status

Portfolio / research demonstration. No production deployment or externally validated performance claims are made here.

## Author

[Karthikeyan Jammalamadaka](https://github.com/jjckarthikeyan-dot)
