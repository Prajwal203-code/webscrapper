# CSV URL Summarizer

A web application that uploads CSV files containing URLs, processes them to extract and summarize content, and provides a downloadable CSV with summaries.

## Features

- **Dark Purple Theme**: Modern UI with neon green accents
- **CSV Upload**: Drag and drop or click to upload CSV files
- **Real-time Processing**: Live progress updates with processing logs
- **URL Summarization**: Automatically scrapes and summarizes content from URLs
- **Download Results**: Get processed CSV with summaries

## UI States

1. **Initial State**: Grey upload button asking for CSV upload
2. **Upload Success**: Green buttons with file name display
3. **Processing**: Real-time logs showing progress with loading spinner
4. **Download Ready**: Success message with download button

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000`

## CSV Format

Your CSV file should have a column named `url` (case-insensitive). The application will also recognize:
- `URL`
- `Url` 
- `link`
- `Link`
- `website`
- `Website`

Example CSV:
```csv
url
https://example.com
https://httpbin.org
https://jsonplaceholder.typicode.com
```

## How It Works

1. **Upload**: Select a CSV file with URLs
2. **Processing**: The application:
   - Reads each URL from the CSV
   - Scrapes the webpage content
   - Generates AI-powered summaries using Hugging Face transformers
   - Updates progress in real-time
3. **Download**: Get the processed CSV with original URLs and their summaries

## Technical Details

- **Backend**: Flask with background processing
- **AI Model**: Facebook BART for text summarization
- **Web Scraping**: BeautifulSoup for content extraction
- **Real-time Updates**: Polling-based status updates
- **File Handling**: Secure file upload and processing

## Requirements

- Python 3.8+
- Internet connection (for AI model download and web scraping)
- Modern web browser

## Notes

- The AI model will be downloaded on first run (may take a few minutes)
- Processing time depends on the number of URLs and their response times
- Large CSV files may take longer to process
- Some websites may block automated requests
