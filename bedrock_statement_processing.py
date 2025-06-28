import boto3
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from botocore.exceptions import ClientError
from typing import List, Optional
from pathlib import Path
import os
import time
import random
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from textractor.data.constants import TextractFeatures
from textractor import Textractor
from textractor.data.html_linearization_config import HTMLLinearizationConfig
import json
import pandas as pd
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Constants
BUCKET_NAME = os.environ.get('BUCKET_NAME')
PDF_BUCKET_PREFIX = os.environ.get('PDF_BUCKET_PREFIX')
OUTPUT_BUCKET_PREFIX = os.environ.get('OUTPUT_BUCKET_PREFIX')
PRODUCT_OUTPUT_FORMAT = os.environ.get('PRODUCT_OUTPUT_FORMAT', 'html')
MODEL_ID = os.environ.get('MODEL_ID')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS'))
TEMPERATURE = float(os.environ.get('TEMPERATURE'))
MODEL_REGION = os.environ.get('MODEL_REGION')
BUCKET_REGION = os.environ.get('BUCKET_REGION')
AWS_REGION = os.environ.get('AWS_REGION')
REGION = os.environ.get('REGION')

def setup_logging():
    """Configure logging with timestamp and appropriate format"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/statement_processing_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('StatementProcessor')

# System prompts for credit card statements
SYSTEM_PROMPT_METADATA = """
<Task>
You are an expert in credit card statement reader, your task is extract key details from given credit card statement.
You need to put your thinking process in the <Thinking> tag.
You must think step by step as below:
1. I need to read and extract all information from provided statement_metadata, put them in <thinking> tag 
2. from all information extracted I need to extract key details from <Output Requirements>
3. Double check the output make sure that above key details extracted are correct and factual.
4. Output key details make sure that meet the requirements in <Output Requirements> tag
5. Compare your output with your output in <Thinking> tag, give your confidence score (1 to 5, 1 is least confidence, 5 is the most confidence)
   about your output
<Task>

<Thinking>

</Thinking>

<Output Requirements>
card_issuer: (e.g., ANZ, CBA, Westpac, NAB, etc.)
card_type: (e.g., Gold, Platinum, Standard, etc.)
account_number: (masked or partial account number)
statement_period: (billing period dates)
statement_date: (date when statement was generated)
payment_due_date: (payment due date)
previous_balance: (balance from previous statement)
payments_credits: (total payments and credits)
purchases_charges: (total purchases and charges)
cash_advances: (total cash advances)
fees_charges: (total fees and charges)
interest_charges: (total interest charges)
new_balance: (new balance after all transactions)
minimum_payment: (minimum payment due)
credit_limit: (total credit limit)
available_credit: (available credit remaining)
</Output Requirements>

Please extract all key details from given credit card statement. 
statement_metadata:
{statement_metadata}
"""

SYSTEM_PROMPT_TRANSACTIONS = """
<Task>
You are an expert in credit card statement reader, your task is extract transaction details from given credit card statement.
You need to put your thinking process in the <Thinking> tag.
You must think step by step as below:
1. I need to read and extract all information from provided transaction_details, put them in <thinking> tag 
2. from all information extracted I need to extract key details from <Output Requirements>
3. Double check the output make sure that above key details extracted are correct and factual.
4. Output key details make sure that meet the requirements in <Output Requirements> tag
5. Compare your output with your output in <Thinking> tag, give your confidence score (1 to 5, 1 is least confidence, 5 is the most confidence)
   about your output
<Task>

<Thinking>

</Thinking>

<Output Requirements>
card_issuer: (the name of the bank or issuer as shown in the statement, e.g., ANZ, CBA, Westpac, NAB, etc.)
transaction_details: [transaction_date,
          merchant_name,
          transaction_description,
          transaction_amount,
          transaction_type, (PURCHASE, PAYMENT, CASH_ADVANCE, FEE, INTEREST, etc.)
          category, (FOOD, TRAVEL, SHOPPING, UTILITIES, etc.)]
confidence_score:
# For any numeric field you cannot determine, output null (not <UNKNOWN> or a string)
<Output Requirements>

Please extract all key details from given credit card statement. 
transaction_details table or text:
{transaction_details}
"""

unknown = " ,output <UNKNOWN> only if you cannot find it"

# Pydantic models for credit card statements
class Transaction(BaseModel):
    transaction_date: str = Field(description="Date of the transaction")
    merchant_name: str = Field(description="Name of the merchant or vendor")
    transaction_description: str = Field(description="Description of the transaction")
    transaction_amount: float = Field(description="Amount of the transaction")
    transaction_type: str = Field(description="Type of transaction (PURCHASE, PAYMENT, CASH_ADVANCE, FEE, INTEREST, etc.)")
    category: str = Field(description="Category of the transaction (FOOD, TRAVEL, SHOPPING, UTILITIES, etc.)")
    reference_number: str = Field(description="Reference or transaction number")

class StatementModelMetadata(BaseModel):
    thinking_process_metadata: str = Field(description="Description of the thought process or reasoning")
    confidence_score: int = Field(description="Confidence score of the output")
    card_issuer: str = Field(description="Card issuer (Visa, Mastercard, American Express, etc.)" + unknown)
    card_type: str = Field(description="Card type (Gold, Platinum, Standard, etc.)" + unknown)
    account_number: str = Field(description="Account number (masked or partial)" + unknown)
    statement_period: str = Field(description="Billing period dates" + unknown)
    statement_date: str = Field(description="Date when statement was generated, format DD-MM-YYYY" + unknown)
    payment_due_date: str = Field(description="Payment due date, format DD-MM-YYYY" + unknown)
    previous_balance: Optional[float] = Field(description="Balance from previous statement")
    payments_credits: Optional[float] = Field(description="Total payments and credits")
    purchases_charges: Optional[float] = Field(description="Total purchases and charges")
    cash_advances: Optional[float] = Field(description="Total cash advances")
    fees_charges: Optional[float] = Field(description="Total fees and charges")
    interest_charges: Optional[float] = Field(description="Total interest charges")
    new_balance: Optional[float] = Field(description="New balance after all transactions")
    minimum_payment: Optional[float] = Field(description="Minimum payment due")
    credit_limit: Optional[float] = Field(description="Total credit limit")
    available_credit: Optional[float] = Field(description="Available credit remaining")

class StatementModelTransactions(BaseModel):
    thinking_process_transactions: str = Field(description="Section containing thinking process information")
    transaction_details: List[Transaction] = Field(default=[], description="List of transactions in the statement")

class StatementProcessor:
    def __init__(self, bucket_name: str, pdf_prefix: str, output_prefix: str, model_id: str):
        self.bucket_name = bucket_name
        self.pdf_prefix = pdf_prefix
        self.output_prefix = output_prefix
        self.model_id = model_id
        self.llm = self._initialize_llm()
        self.logger = setup_logging()
        self.logger.info("StatementProcessor initialized with bucket: %s", bucket_name)
        
    def _initialize_llm(self):
        return ChatBedrock(
            credentials_profile_name="default",
            model_id=self.model_id,
            model_kwargs={
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
        )

    def pdf_to_markdown_html(self, s3_path, transaction_output_format='html'):
        self.logger.info(f"Converting PDF to {transaction_output_format}: %s", s3_path)
        input_pair = {}
        extractor = Textractor(profile_name="default", region_name=BUCKET_REGION)
        config = HTMLLinearizationConfig()
        config.table_cell_header_prefix = "<td>"
        config.table_cell_header_suffix = "</td>"
        
        document = extractor.start_document_analysis(
            file_source=s3_path,
            features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES, TextractFeatures.FORMS],
        )

        # Aggregate all tables from all pages
        if len(document.tables) > 0:
            self.logger.info(f"Found {len(document.tables)} tables in document.")
            for i, table in enumerate(document.tables):
                if hasattr(table, 'nrows') and hasattr(table, 'ncols'):
                    self.logger.info(f"Table {i+1} size: {table.nrows} rows x {table.ncols} columns")
                else:
                    self.logger.info(f"Table {i+1} summary: {str(table)[:100]}")
            if transaction_output_format == 'html':
                all_tables = "".join([table.to_html() for table in document.tables])
                transaction_table = all_tables
            elif transaction_output_format == 'markdown':
                all_tables = "\n\n".join([table.to_markdown() for table in document.tables])
                transaction_table = all_tables
            else:
                self.logger.warning(f"Unknown transaction_output_format: {transaction_output_format}. Defaulting to HTML.")
                all_tables = "".join([table.to_html() for table in document.tables])
                transaction_table = all_tables
            self.logger.info(f"Total transaction_table length: {len(transaction_table)} characters")
            self.logger.info(f"First 500 chars of transaction_table: {transaction_table[:500]}")
            self.logger.info(f"Last 500 chars of transaction_table: {transaction_table[-500:]}")
        else:
            self.logger.warning("No tables found in document. Falling back to lines of text.")
            # Fallback: extract all lines of text
            all_lines = []
            for page in document.pages:
                if hasattr(page, 'lines'):
                    all_lines.extend([line.text for line in page.lines])
            transaction_table = "\n".join(all_lines)
            self.logger.info(f"First 500 chars of transaction_table (lines fallback): {transaction_table[:500]}")
        
        # Aggregate all key-values from all pages (if needed)
        if hasattr(document.key_values, 'get_text'):
            statement_metadata = document.key_values.get_text()
        else:
            statement_metadata = "\n".join([kv.get_text() for kv in document.key_values])
        input_pair['statement_metadata'] = statement_metadata
        input_pair['transaction_details'] = transaction_table
        return input_pair

    def process_transaction_chunks(self, transaction_details, transaction_prompt):
        """Process large transaction tables by splitting into chunks"""
        chunks = self.split_html_table_into_chunks(transaction_details, max_chunk_size=5000)
        self.logger.info(f"Split transaction table into {len(chunks)} chunks")
        
        all_transactions = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} characters")
            try:
                chunk_prompt = transaction_prompt.format(transaction_details=chunk)
                chunk_response = self.make_api_call(chunk_prompt, StatementModelTransactions)
                chunk_transactions = chunk_response.transaction_details
                all_transactions.extend(chunk_transactions)
                self.logger.info(f"Chunk {i+1} extracted {len(chunk_transactions)} transactions")
                time.sleep(2)  # Brief pause between chunks
            except Exception as e:
                self.logger.error(f"Failed to process chunk {i+1}: {str(e)}")
                continue
        
        return all_transactions

    def split_html_table_into_chunks(self, html_content, max_chunk_size=5000):
        """Split HTML table into smaller chunks while preserving table structure"""
        chunks = []
        
        # Find table rows
        import re
        rows = re.findall(r'<tr[^>]*>.*?</tr>', html_content, re.DOTALL | re.IGNORECASE)
        
        if not rows:
            # If no table rows found, split by length
            return [html_content[i:i+max_chunk_size] for i in range(0, len(html_content), max_chunk_size)]
        
        # Group rows into chunks
        current_chunk = "<table>"
        header_row = ""
        
        # Try to find header row (first row)
        if rows:
            first_row = rows[0]
            if 'th>' in first_row.lower() or any(header_word in first_row.lower() for header_word in ['date', 'amount', 'description', 'merchant']):
                header_row = first_row
                rows = rows[1:]  # Remove header from processing
        
        for row in rows:
            # Check if adding this row would exceed chunk size
            potential_chunk = current_chunk + header_row + row + "</table>"
            
            if len(potential_chunk) > max_chunk_size and current_chunk != "<table>":
                # Finalize current chunk
                chunks.append(current_chunk + header_row + "</table>")
                current_chunk = "<table>"
            
            current_chunk += row
        
        # Add the last chunk
        if current_chunk != "<table>":
            chunks.append(current_chunk + header_row + "</table>")
        
        return chunks

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def make_api_call(self, prompt, output_schema):
        self.logger.debug("Making API call with schema: %s", output_schema.__name__)
        model = self.llm.with_structured_output(output_schema)
        return model.invoke(prompt)

    def process_statements(self):
        self.logger.info("Starting statement processing")
        files = self.list_pdf_files()
        self.logger.info("Found %d PDF files to process", len(files))
        
        model_inputs = []
        self.logger.info("Converting PDFs to markdown/HTML")
        for file in tqdm(files, desc="Converting PDFs", unit="file"):
            model_inputs.append(self.pdf_to_markdown_html(file, PRODUCT_OUTPUT_FORMAT))

        all_response = []
        failed_inputs = []
        metadata_prompt = PromptTemplate(
            input_variables=["statement_metadata"],
            template=SYSTEM_PROMPT_METADATA
        )
        transaction_prompt = PromptTemplate(
            input_variables=["transaction_details"],
            template=SYSTEM_PROMPT_TRANSACTIONS
        )

        self.logger.info("Processing statement metadata and transaction details")
        for input in tqdm(model_inputs, desc="Processing statements", unit="statement"):
            try:
                self.logger.debug("Processing input batch")
                time.sleep(random.uniform(1, 3))
                metadata = metadata_prompt.format(statement_metadata=input['statement_metadata'])
                
                # Process metadata first
                metadata_response = self.make_api_call(metadata, StatementModelMetadata)
                
                # Handle large transaction tables by chunking
                transaction_details = input['transaction_details']
                self.logger.info(f"Processing transaction data with {len(transaction_details)} characters")
                
                # If transaction data is large (>6000 chars), split into chunks
                if len(transaction_details) > 6000:
                    self.logger.info("Large transaction table detected, using chunking strategy")
                    all_transactions = self.process_transaction_chunks(transaction_details, transaction_prompt)
                else:
                    transactions = transaction_prompt.format(transaction_details=transaction_details)
                    transaction_response = self.make_api_call(transactions, StatementModelTransactions)
                    all_transactions = transaction_response.transaction_details
                
                self.logger.info(f"Total transactions extracted: {len(all_transactions)}")
                
                # Merge responses
                metadata_dict = json.loads(metadata_response.model_dump_json())
                merged_response = {
                    **metadata_dict,
                    'transaction_details': all_transactions,
                    'thinking_process_transactions': f"Processed {len(all_transactions)} transactions from transaction table"
                }
                all_response.append(merged_response)
                self.logger.info("Successfully processed input batch")
                
                time.sleep(5)
                
            except Exception as e:
                self.logger.error("Failed to process input: %s", str(e), exc_info=True)
                failed_inputs.append(input)
                time.sleep(10)

        if failed_inputs:
            self.logger.warning("Failed to process %d inputs", len(failed_inputs))

        saved_files = self.save_json_responses(files, all_response)
        uploaded_files = self.upload_json_to_s3('output_json')
        
        return self.create_dataframe(all_response, files)

    def list_pdf_files(self) -> List[str]:
        if not self.bucket_name:
            raise ValueError("Bucket name cannot be empty")
        
        s3_client = boto3.client('s3', region_name=BUCKET_REGION)
        pdf_files = []
        
        try:
            s3_client.head_bucket(Bucket=self.bucket_name)
            paginator = s3_client.get_paginator('list_objects_v2')
            operation_parameters = {
                'Bucket': self.bucket_name,
                'Prefix': self.pdf_prefix
            }
            
            page_iterator = paginator.paginate(**operation_parameters)
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith('.pdf'):
                            pdf_files.append(f"s3://{self.bucket_name}/{key}")
            
            self.logger.info("Found %d PDF files in bucket", len(pdf_files))
            if len(pdf_files) > 0:
                return pdf_files
            else:
                raise ValueError("No PDF files found in the specified bucket")
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            self.logger.error("S3 client error: %s", error_code)
            if error_code in ['NoSuchBucket', 'AccessDenied']:
                raise
            else:
                raise
        except Exception as e:
            self.logger.error("Unexpected error listing PDF files: %s", str(e))
            raise RuntimeError(f"Unexpected error: {str(e)}") from e

    def save_json_responses(self, files, all_response, output_dir='output_json'):
        self.logger.info("Saving JSON responses to directory: %s", output_dir)
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            saved_files = []

            for file_name, response in tqdm(zip(files, all_response), desc="Saving JSON responses", unit="file", total=len(files)):
                try:
                    base_name = file_name.split("/")[-1].split('.')[0]
                    output_name = f"{base_name}.json"
                    output_path = os.path.join(output_dir, output_name)
                    
                    self.logger.debug("Attempting to save: %s", output_name)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(response, f,
                                indent=4,
                                ensure_ascii=False,
                                sort_keys=True)
                    
                    if os.path.exists(output_path):
                        self.logger.info("Successfully saved: %s", output_name)
                        saved_files.append(output_path)
                    else:
                        self.logger.warning("File not found after saving: %s", output_name)
                        
                except Exception as e:
                    self.logger.error("Error saving %s: %s", base_name, str(e))
            
            return saved_files
        
        except Exception as e:
            self.logger.error("Error creating directory %s: %s", output_dir, str(e))
            return []

    def upload_json_to_s3(self, local_folder: str, file_pattern: str = '*.json') -> List[str]:
        self.logger.info("Starting S3 upload from folder: %s", local_folder)
        
        if not self.bucket_name:
            raise ValueError("Bucket name cannot be empty")
        
        if not local_folder:
            raise ValueError("Local folder path cannot be empty")
        
        s3_client = boto3.client('s3', region_name=BUCKET_REGION)
        uploaded_files = []
        
        try:
            s3_client.head_bucket(Bucket=self.bucket_name)
            
            folder_path = Path(local_folder)
            if not folder_path.exists():
                raise ValueError(f"Local folder not found: {local_folder}")
            
            json_files = list(folder_path.glob(file_pattern))
            
            if not json_files:
                self.logger.warning("No JSON files found in %s", local_folder)
                return uploaded_files
            
            for json_file in tqdm(json_files, desc="Uploading to S3", unit="file"):
                try:
                    s3_key = f"{self.output_prefix.rstrip('/')}/{json_file.name}"
                    self.logger.debug("Uploading %s to s3://%s/%s", json_file.name, 
                                    self.bucket_name, s3_key)
                    
                    s3_client.upload_file(
                        str(json_file),
                        self.bucket_name,
                        s3_key,
                        ExtraArgs={
                            'ContentType': 'application/json'
                        }
                    )
                    
                    uploaded_files.append(f"s3://{self.bucket_name}/{s3_key}")
                    self.logger.info("Successfully uploaded: %s", json_file.name)
                    
                except ClientError as e:
                    self.logger.error("Error uploading %s: %s", json_file.name, 
                                    e.response['Error']['Message'])
                except Exception as e:
                    self.logger.error("Unexpected error uploading %s: %s", 
                                    json_file.name, str(e))
            
            return uploaded_files
            
        except ClientError as e:
            self.logger.error("S3 client error: %s", e.response['Error']['Message'])
            raise
        except Exception as e:
            self.logger.error("Unexpected error during S3 upload: %s", str(e))
            raise

    def detect_bank_name_from_text(self, text):
        text = text.lower()
        if 'anz' in text:
            return 'anz'
        elif 'commonwealth' in text or 'cba' in text:
            return 'cba'
        elif 'nab' in text or 'national australia bank' in text:
            return 'nab'
        elif 'westpac' in text:
            return 'westpac'
        else:
            return 'unknown'

    def create_dataframe(self, all_response, files=None):
        self.logger.info("Creating DataFrame from responses")
        try:
            # Flatten the nested structure for DataFrame creation
            flattened_data = []
            for idx, response in enumerate(all_response):
                # Prefer card_issuer from model output, fallback to metadata detection
                card_issuer = response.get('card_issuer', '')
                if not card_issuer:
                    statement_metadata = response.get('statement_metadata', '')
                    card_issuer = self.detect_bank_name_from_text(statement_metadata)
                # Extract metadata
                metadata = {
                    'confidence_score': response.get('confidence_score', 0),
                    'card_issuer': card_issuer
                }
                # Extract transactions
                transactions = response.get('transaction_details', [])
                if transactions:
                    for transaction in transactions:
                        row = metadata.copy()
                        # Handle both dict and Pydantic model objects
                        if hasattr(transaction, 'model_dump'):
                            # Pydantic model - convert to dict
                            transaction_dict = transaction.model_dump()
                        else:
                            # Already a dict
                            transaction_dict = transaction
                        
                        row.update({
                            'transaction_date': transaction_dict.get('transaction_date', ''),
                            'merchant_name': transaction_dict.get('merchant_name', ''),
                            'transaction_description': transaction_dict.get('transaction_description', ''),
                            'transaction_amount': transaction_dict.get('transaction_amount', 0.0),
                            'transaction_type': transaction_dict.get('transaction_type', ''),
                            'category': transaction_dict.get('category', '')
                        })
                        flattened_data.append(row)
                else:
                    # If no transactions, still include the metadata
                    flattened_data.append(metadata)
            df = pd.DataFrame(flattened_data)
            # Filter out records with confidence_score lower than 1 (lowered from 2 to capture more data)
            df = df[df['confidence_score'] >= 1]
            # Only keep the columns needed for credit card statement analysis
            columns_to_keep = [
                'confidence_score',
                'card_issuer',
                'transaction_date',
                'merchant_name',
                'transaction_description',
                'transaction_amount',
                'transaction_type',
                'category'
            ]
            # Only keep columns that exist in the DataFrame
            df = df[[col for col in columns_to_keep if col in df.columns]]
            df.to_csv('outputs.csv', index=False)
            self.logger.info("DataFrame created and saved to outputs.csv")
            return df
            
        except Exception as e:
            self.logger.error("Error creating DataFrame: %s", str(e))
            raise

def get_current_model_id():
    """Return the current MODEL_ID from the environment."""
    return MODEL_ID

def main():
    try:
        print(f"[INFO] Using MODEL_ID: {get_current_model_id()}")
        processor = StatementProcessor(
            bucket_name=BUCKET_NAME,
            pdf_prefix=PDF_BUCKET_PREFIX,
            output_prefix=OUTPUT_BUCKET_PREFIX,
            model_id=MODEL_ID
        )
        
        result_df = processor.process_statements()
        print("Statement processing completed successfully!")
        print(f"Processed {len(result_df)} records")
        
    except Exception as e:
        logging.error("Application failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main() 