# Credit Card Statement Processing System

An automated system for processing credit card statements using AWS Bedrock, Amazon Textract, and LangChain. This system extracts information from PDF credit card statements, processes them using AI, and generates structured data output.

## Features

- PDF credit card statement processing using Amazon Textract for accurate data extraction
- AI-powered information extraction using AWS Bedrock with Claude model
- Automatic metadata extraction (card issuer, account number, statement dates, balances)
- Transaction details extraction with structured output
- Support for both HTML and Markdown output formats
- S3 integration for file storage and retrieval
- Structured JSON output generation with confidence scoring
- DataFrame creation with CSV export for data analysis
- Comprehensive logging system with timestamped logs
- Progress tracking with tqdm for long-running operations
- Robust error handling and retry mechanisms
- Configurable batch processing with rate limiting

## Prerequisites

- Python 3.x
- AWS Account with appropriate permissions
  - Amazon Textract access
  - AWS Bedrock access (Claude model)
  - S3 bucket permissions
- AWS credentials configured locally
- S3 bucket for storing credit card statements and output

## Environment Variables

Create a `.env` file with the following configurations:

```
BUCKET_NAME=your-bucket-name          # S3 bucket for storing statements and output
PDF_BUCKET_PREFIX=statements          # Prefix for input PDF files
BUCKET_REGION=ap-southeast-2          # S3 bucket region
OUTPUT_BUCKET_PREFIX=output_json      # Prefix for processed JSON output
PRODUCT_OUTPUT_FORMAT=markdown        # Format for transaction tables (markdown or html)
MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0  # AWS Bedrock model ID
MAX_TOKENS=10000                      # Maximum tokens for model response
TEMPERATURE=0.0                       # Model temperature (0.0 for consistent output)
REGION=us-east-1                      # AWS region
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Configure your AWS credentials and environment variables
2. Place your PDF credit card statements in the configured S3 bucket under the specified bucket (BUCKET_NAME) and prefix (PDF_BUCKET_PREFIX)
3. Run the processor:
   ```
   python bedrock_statement_processing.py
   ```

## Output

The system generates three types of output:

1. JSON files
   - Stored in the 'output_json' directory locally
   - Uploaded to S3 under OUTPUT_BUCKET_PREFIX
   - Contains structured statement data with confidence scores

2. CSV file (outputs.csv)
   - Consolidated view of all processed statements
   - Includes both metadata and transaction details
   - Suitable for data analysis and reporting

3. Log file
   - Timestamped log file (statement_processing_YYYYMMDD_HHMMSS.log) stored in logs folder
   - Contains detailed processing information
   - Tracks successful operations and any errors

## Error Handling

- Automatic retries for API calls with exponential backoff
- Detailed error logging for failed operations
- Tracking of failed inputs for potential reprocessing
- Rate limiting to prevent API throttling

## Processing Features

- Metadata extraction including:
  - Card issuer and type information
  - Account numbers and statement periods
  - Balance information (previous, new, available credit)
  - Payment due dates and minimum payments
  - Credit limits and fees

- Transaction details extraction:
  - Transaction dates and merchant names
  - Transaction descriptions and amounts
  - Transaction types (PURCHASE, PAYMENT, CASH_ADVANCE, etc.)
  - Categories (FOOD, TRAVEL, SHOPPING, etc.)
  - Reference numbers
  - Structured output in table format

## Supported Credit Card Statement Formats

The system is designed to handle various credit card statement formats from major issuers including:
- Visa, Mastercard, American Express
- Various card types (Standard, Gold, Platinum, etc.)
- Different statement layouts and table structures
- Multiple transaction types and categories