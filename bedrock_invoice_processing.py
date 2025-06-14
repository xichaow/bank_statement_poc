import boto3
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from botocore.exceptions import ClientError
from typing import List
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
PRODUCT_OUTPUT_FORMAT = os.environ.get('PRODUCT_OUTPUT_FORMAT')
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
    log_file = f'logs/invoice_processing_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('InvoiceProcessor')

# System prompts
SYSTEM_PROMPT_METADATA = """
<Task>
You are an expert in invoice reader, your task is extract key details from given invoice.
You need to put your thinking process in the <Thinking> tag.
You must think step by step as below:
1. I need to read and extract all information from provided invoice_metadata, put them in <thinking> tag 
2. from all information extracted I need to extract key details from <Output Requirements>
3. Double check the output make sure that above key details extracted are correct and factual, meet Australian Standards.
4. Output key details make sure that meet the requirements in <Output Requirements> tag
5. Compare your output with your output in <Thinking> tag, give your confidence score (1 to 5, 1 is least confidence, 5 is the most confidence)
   about your output
<Task\>

<Thinking>

<Thinking\>

<Output Requirements>
vendor_name:
invoice_id: 
invoice_to:
invoice_amount:
invoice_date:
total_tax(GST):
invoice_description:
invoice_due_date:
<Output Requirements\>

Please extract all key details from given invoice. 
invoice_metadata:
{invoice_metadata}
"""

SYSTEM_PROMPT_PRODUCT = """
<Task>
You are an expert in invoice reader, your task is extract key details from given invoice.
You need to put your thinking process in the <Thinking> tag.
You must think step by step as below:
1. I need to read and extract all information from provided product_details, put them in <thinking> tag 
2. from all information extracted I need to extract key details from <Output Requirements>
3. Double check the output make sure that above key details extracted are correct and factual, meet Australian Standards.
4. Output key details make sure that meet the requirements in <Output Requirements> tag
5. Compare your output with your output in <Thinking> tag, give your confidence score (1 to 5, 1 is least confidence, 5 is the most confidence)
   about your output
<Task\>

<Thinking>

<Thinking\>

<Output Requirements>
product_details: [product_code,
          product_name,
          price_of_product_excluding_tax,
          tax_of_product,
          number_of_unit,
          subtotal_amount_of_product]
confidence_score:
<Output Requirements\>

Please extract all key details from given invoice. 
product_details table:
{product_details}
"""

unknown = " ,output <UNKNOWN> only if you cannot find it"

# Pydantic models
class Product(BaseModel):
    product_name: str = Field(description="Name of the product")
    product_code: str = Field(description="Individual product code")
    price_of_product_excluding_tax: float = Field(description="Price of the product before tax")
    tax_of_product: float = Field(description="Tax amount for the product")
    number_of_unit: int = Field(description="Quantity of product units ordered only")
    subtotal_amount_of_product: float = Field(description="subtotal amount for this product product inlcuding tax")

class InvoiceModelMetadata(BaseModel):
    thinking_process_metadata: str = Field(description="Description of the thought process or reasoning")
    confidence_score: int = Field(description="Confidence score of the output")
    vendor_name: str = Field(description="Name of the vendor or supplier or invoice issuer" + unknown)
    invoice_id: str = Field(description="Unique identifier for the invoice" + unknown)
    invoice_to: str = Field(description="Name of the recipient or customer" + unknown)
    invoice_amount: float = Field(description="Total amount of the invoice")
    invoice_date: str = Field(description="Date when the invoice was issued， date format DD-MM-YYYY" + unknown)
    total_tax: float = Field(description="Total tax(GST) amount for the entire invoice")
    invoice_description: str = Field(description="Detailed description of the invoice" + unknown)
    invoice_due_date: str = Field(description="Date when the payment is due, date format DD-MM-YYYY" + unknown)
    abn: str = Field(description="Australian Business Number is a unique 11-digit number that identifies your business or organisation to the government and community." + unknown)

class InvoiceModelProduct(BaseModel):
    thinking_process_product_details: str = Field(description="Section containing thinking process information")
    product_details: List[Product] = Field(default=[], description="List of products included in the invoice")

class InvoiceProcessor:
    def __init__(self, bucket_name: str, pdf_prefix: str, output_prefix: str, model_id: str):
        self.bucket_name = bucket_name
        self.pdf_prefix = pdf_prefix
        self.output_prefix = output_prefix
        self.model_id = model_id
        self.llm = self._initialize_llm()
        self.logger = setup_logging()
        self.logger.info("InvoiceProcessor initialized with bucket: %s", bucket_name)
        
    def _initialize_llm(self):
        return ChatBedrock(
            credentials_profile_name="default",
            model_id=self.model_id,
            model_kwargs={
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
        )

    def pdf_to_markdown_html(self, s3_path, product_output_format='html'):
        self.logger.info(f"Converting PDF to {product_output_format}: %s", s3_path)
        input_pair = {}
        extractor = Textractor(profile_name="default", region_name=BUCKET_REGION)
        config = HTMLLinearizationConfig()
        config.table_cell_header_prefix = "<td>"
        config.table_cell_header_suffix = "</td>"
        
        document = extractor.start_document_analysis(
            file_source=s3_path,
            features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES, TextractFeatures.FORMS],
        )

        if product_output_format == 'html':
            product_table = document.tables.to_html(config)
        elif product_output_format == 'markdown':
            product_table = document.tables.to_markdown()
            
        invoice_metadata = document.key_values.get_text()
        input_pair['invoice_metadata'] = invoice_metadata
        input_pair['product_details'] = product_table
        return input_pair

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def make_api_call(self, prompt, output_schema):
        self.logger.debug("Making API call with schema: %s", output_schema.__name__)
        model = self.llm.with_structured_output(output_schema)
        return model.invoke(prompt)

    def process_invoices(self):
        self.logger.info("Starting invoice processing")
        files = self.list_pdf_files()
        self.logger.info("Found %d PDF files to process", len(files))
        
        model_inputs = []
        self.logger.info("Converting PDFs to markdown/HTML")
        for file in tqdm(files, desc="Converting PDFs", unit="file"):
            model_inputs.append(self.pdf_to_markdown_html(file, PRODUCT_OUTPUT_FORMAT))

        all_response = []
        failed_inputs = []
        metadata_prompt = PromptTemplate(
            input_variables=["invoice_metadata"],
            template=SYSTEM_PROMPT_METADATA
        )
        product_prompt = PromptTemplate(
            input_variables=["product_details"],
            template=SYSTEM_PROMPT_PRODUCT
        )

        self.logger.info("Processing invoice metadata and product details")
        for input in tqdm(model_inputs, desc="Processing invoices", unit="invoice"):
            try:
                self.logger.debug("Processing input batch")
                time.sleep(random.uniform(1, 3))
                metadata = metadata_prompt.format(invoice_metadata=input['invoice_metadata'])
                product = product_prompt.format(product_details=input['product_details'])
                metadata_response = self.make_api_call(metadata, InvoiceModelMetadata)
                product_response = self.make_api_call(product, InvoiceModelProduct)
                merged_response = json.loads(metadata_response.model_dump_json()) | json.loads(product_response.model_dump_json())
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
        
        return self.create_dataframe(all_response)

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
            error_code = e.response.get('Error', {}).get('Code', '')
            self.logger.error("S3 client error: %s", error_code)
            if error_code in ['NoSuchBucket', 'AccessDenied']:
                raise
            else:
                raise
        except Exception as e:
            self.logger.error("Unexpected error in S3 upload: %s", str(e))
            raise RuntimeError(f"Unexpected error: {str(e)}") from e

    def create_dataframe(self, all_response):
        self.logger.info("Creating DataFrame from responses")
        try:
            all_df = pd.DataFrame()
            for response in all_response:
                df = pd.DataFrame([response])
                df = df.explode('product_details')
                all_df = pd.concat([all_df, df], ignore_index=True)
            self.logger.info("Successfully created DataFrame with %d rows", len(all_df))
            all_df.to_csv('outputs.csv')
            return all_df
        except Exception as e:
            self.logger.error("Error creating DataFrame: %s", str(e))
            raise

def main():
    logger = setup_logging()
    logger.info("Starting invoice processing application")
    
    try:
        processor = InvoiceProcessor(
            bucket_name=BUCKET_NAME,
            pdf_prefix=PDF_BUCKET_PREFIX,
            output_prefix=OUTPUT_BUCKET_PREFIX,
            model_id=MODEL_ID
        )
        
        result_df = processor.process_invoices()
        logger.info("Successfully completed invoice processing")
        return result_df
        
    except Exception as e:
        logger.error("Application failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()