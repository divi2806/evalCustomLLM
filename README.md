# Custom LLM QA Evaluation

This Streamlit application provides a comprehensive tool for evaluating Question-Answering (QA) systems, with a specific focus on custom Language Model (LLM) QA systems. It offers functionality to process various file types, analyze QA performance, and visualize results.

## Features

- File Processing: Supports PDF, Excel, Word, Plain Text, CSV, and Google Sheets.
- Text Chunking and Embedding: Uses LangChain for text processing and HuggingFace embeddings.
- QA Evaluation: Compares custom LLM answers against ground truths using multiple metrics.
- Visualization: Generates plots for easy interpretation of results.
- Results Export: Saves detailed evaluation results to a CSV file.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/custom-llm-qa-evaluation.git
   cd custom-llm-qa-evaluation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. For Google Sheets functionality, set up your credentials:
   - Create a service account and download the JSON key file.
   - Rename it to `credentials.json` and place it in the project directory.
   - Add the path to your credentials file in the `read_google_sheet` function.

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

Follow the on-screen instructions to:
1. Select and upload your file (PDF, Excel, Word, Plain Text, CSV, or enter a Google Sheets URL).
2. Upload your questions, ground truths, and custom LLM answers files.
3. View the evaluation results and visualizations.

## Metrics

The application calculates the following metrics:
- Similarity (Cosine Similarity with Ground Truth)
- Exact Match
- Faithfulness
- Relevance
- Hallucination
- QA Accuracy
- Toxicity

## Output

- Detailed metrics for each question-answer pair.
- Aggregate metrics across all questions.
- Visualizations of metric distributions.
- CSV file with detailed results.

## Dependencies

Main dependencies include:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- PyPDF2
- python-docx
- langchain
- scikit-learn
- detoxify
- gspread

For a complete list, see `requirements.txt`.

## Contributing

Contributions to improve the Custom LLM QA Evaluation tool are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on the GitHub repository.
me.md…]()
