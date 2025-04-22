# app.py - Flask web application for AI Finance Accountant Agent
from flask import Flask, render_template, request, jsonify
import base64
import tempfile
import os
import json
import logging
from datetime import datetime, timedelta
import openai
import numpy as np
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("finance_agent_web.log"), logging.StreamHandler()]
)
logger = logging.getLogger("WebFinanceAccountantAgent")

# API keys - replace with your actual keys
OPENAI_API_KEY = " "
FINANCIAL_API_KEY = "1CRymfJDvoY3Ufs5bBCBi2wXnwAhE73M"

# Configuration settings
CONFIG = {
    "rag": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "model": {
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o",
        "speech_to_text": "whisper-1"
    },
    "api": {
        "base_url": "https://api.yourfinanceprovider.com/v1"  # Replace with actual API endpoint
    }
}

class KnowledgeBase:
    """Manages the RAG (Retrieval-Augmented Generation) system"""
    
    def __init__(self, config, openai_api_key):
        self.config = config
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = None
        self.init_knowledge_base()
        
    def init_knowledge_base(self):
        """Initialize the knowledge base with financial domain knowledge"""
        logger.info("Initializing knowledge base...")
        
        # Financial knowledge corpus - in a real system, this would be loaded from files
        financial_docs = [
            "Financial statements include balance sheets, income statements, and cash flow statements.",
            "Accounts receivable represents money owed to a company by its customers.",
            "Accounts payable represents money a company owes to its vendors or suppliers.",
            "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization.",
            "ROI (Return on Investment) is calculated by dividing net profit by the cost of investment.",
            "A general ledger is a complete record of all financial transactions of a company.",
            "Depreciation is the allocation of an asset's cost over its useful life.",
            "Financial ratios help analyze a company's financial health and performance.",
            "Liquidity ratios measure a company's ability to pay short-term obligations.",
            "Solvency ratios measure a company's ability to meet long-term obligations.",
            "Revenue is the total income generated from sales of goods or services.",
            "Expenses are costs incurred in the process of generating revenue.",
            "Net income is calculated by subtracting expenses from revenue.",
            "A fiscal year is a one-year period used for financial reporting.",
            "A budget is a financial plan for a specified period.",
            "Common financial APIs include Plaid, Stripe, Square, and PayPal.",
            "To create an invoice, you need client information, itemized services/products, and payment terms.",
            "To check account balances, use the account API endpoint with proper authentication.",
            "To process payments, you need payment amount, source, destination, and authorization.",
            "Financial compliance requires adherence to regulations like GAAP, IFRS, and Sarbanes-Oxley."
        ]
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        texts = text_splitter.create_documents(financial_docs)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        logger.info("Knowledge base initialized successfully")
    
    def query_knowledge_base(self, query: str, k: int = 3):
        """Query the knowledge base to retrieve relevant context"""
        logger.info(f"Querying knowledge base with: {query}")
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

class CommandProcessor:
    """Processes and interprets commands using LLM"""
    
    def __init__(self, config, knowledge_base, openai_api_key):
        self.config = config
        self.knowledge_base = knowledge_base
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        
    def process_audio(self, audio_file_path):
        """Process audio file to text using OpenAI's Whisper"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.config["speech_to_text"],
                    file=audio_file
                )
            return transcription.text
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None
            
    def process_text(self, text):
        """Process text input directly"""
        return text
        
    def interpret_command(self, input_text):
        """Interpret the command using GPT and RAG"""
        logger.info(f"Interpreting command: {input_text}")
        
        # Get relevant context from knowledge base
        context = self.knowledge_base.query_knowledge_base(input_text)
        context_text = "\n".join(context)
        
        # Define keywords for different operations to help with simple pattern matching
        operation_keywords = {
            "check_balance": ["balance", "account", "how much", "available funds", "check my account"],
            "create_invoice": ["invoice", "bill", "create invoice", "new invoice", "charge"],
            "process_payment": ["payment", "pay", "transfer", "send money", "process payment"],
            "generate_report": ["report", "statement", "generate report", "financial report"]
        }
        
        # Simple pattern matching to suggest operation type
        detected_operation = "other"
        for op, keywords in operation_keywords.items():
            if any(keyword in input_text.lower() for keyword in keywords):
                detected_operation = op
                break
        
        # Prepare the prompt with context and suggested operation
        prompt = f"""
        You are an AI finance accountant assistant. Interpret the following command and extract structured information.
        
        Relevant financial knowledge:
        {context_text}
        
        Command: "{input_text}"
        
        Initial operation suggestion based on keywords: {detected_operation}
        
        Identify the financial operation requested. Choose ONE of the following:
        - check_balance: For checking account balances or financial status
        - create_invoice: For creating or requesting new invoices
        - process_payment: For making payments or transfers
        - generate_report: For generating financial reports or statements
        - other: For general financial questions or requests that don't fit above categories
        
        For each operation type, extract these specific parameters:
        - check_balance: account_id, account_type
        - create_invoice: client_name, amount, due_date, items
        - process_payment: recipient, amount, payment_method, payment_date
        - generate_report: report_type, period, format
        - other: query (the general question being asked)
        
        Format your response as a valid JSON object with these fields:
        - operation: the type of operation requested (one of the above)
        - parameters: a dictionary of parameters needed for the operation
        - confidence: your confidence score (0-1) in this interpretation
        
        Important:
        1. Be very specific about the operation type based on the command
        2. Extract as many parameters as possible from the command
        3. DO NOT add parameters that weren't mentioned in the command
        4. Make sure your response is a valid JSON object
        5. If you're uncertain, use the "other" operation type
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            logger.info(f"Raw interpretation response: {content}")
            
            try:
                interpretation = json.loads(content)
                
                # Validate the interpretation
                if "operation" not in interpretation:
                    interpretation["operation"] = detected_operation
                
                if "parameters" not in interpretation:
                    interpretation["parameters"] = {}
                
                if "confidence" not in interpretation:
                    interpretation["confidence"] = 0.7
                    
                logger.info(f"Command interpreted as: {interpretation}")
                return interpretation
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {str(je)}")
                # Fall back to simple pattern matching result
                return {
                    "operation": detected_operation,
                    "parameters": {"query": input_text},
                    "confidence": 0.5
                }
                
        except Exception as e:
            logger.error(f"Error interpreting command: {str(e)}")
            # Return a fallback interpretation to prevent errors
            return {
                "operation": detected_operation,
                "parameters": {"query": input_text},
                "confidence": 0.3,
                "error": str(e)
            }

class APIConnector:
    """Handles connections to financial APIs"""
    
    def __init__(self, config, financial_api_key):
        self.config = config
        self.base_url = config["base_url"]
        self.financial_api_key = financial_api_key
        self.headers = {
            "Authorization": f"Bearer {self.financial_api_key}",
            "Content-Type": "application/json"
        }
        # Sample data for demo purposes
        self.accounts = {
            "main": {
                "account_id": "main",
                "balance": 15250.75,
                "currency": "USD",
                "available": 14980.50,
                "last_updated": "2025-04-21T08:30:45Z"
            },
            "savings": {
                "account_id": "savings",
                "balance": 42500.00,
                "currency": "USD",
                "available": 42500.00,
                "last_updated": "2025-04-21T08:30:45Z"
            },
            "business": {
                "account_id": "business",
                "balance": 87325.18,
                "currency": "USD",
                "available": 85450.22,
                "last_updated": "2025-04-21T08:30:45Z"
            }
        }
    
    def execute_operation(self, operation, parameters):
        """Execute the financial operation based on the interpreted command"""
        logger.info(f"Executing operation: {operation} with parameters: {parameters}")
        
        if operation == "check_balance":
            return self.check_balance(parameters.get("account_id", "main"))
            
        elif operation == "create_invoice":
            return self.create_invoice(parameters)
            
        elif operation == "process_payment":
            return self.process_payment(parameters)
            
        elif operation == "generate_report":
            return self.generate_report(parameters)
            
        else:
            # Handle generic queries with a comprehensive response
            return self.handle_generic_query(parameters.get("query", ""))
    
    def check_balance(self, account_id):
        """Check balance for a specific account"""
        logger.info(f"Checking balance for account: {account_id}")
        
        # Use account data from our mock database
        if account_id.lower() in self.accounts:
            return self.accounts[account_id.lower()]
        else:
            # Default to main account if not found
            return self.accounts["main"]
    
    def create_invoice(self, invoice_data):
        """Create a new invoice"""
        logger.info(f"Creating invoice for client: {invoice_data.get('client_name')}")
        
        client_name = invoice_data.get("client_name", "Default Client")
        amount = invoice_data.get("amount", 0)
        
        # Generate due date if not provided (30 days from today)
        due_date = invoice_data.get("due_date", "") 
        if not due_date:
            due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        items = invoice_data.get("items", [{"description": "Services", "amount": amount}])
        
        return {
            "invoice_id": f"INV-{datetime.now().strftime('%Y%m%d')}-001",
            "client_name": client_name,
            "amount": amount,
            "items": items,
            "status": "created",
            "due_date": due_date,
            "created_at": datetime.now().isoformat()
        }
    
    def process_payment(self, payment_data):
        """Process a payment"""
        logger.info(f"Processing payment of {payment_data.get('amount')} to {payment_data.get('recipient')}")
        
        amount = payment_data.get("amount", 0)
        recipient = payment_data.get("recipient", "Default Recipient")
        payment_method = payment_data.get("payment_method", "bank_transfer")
        payment_date = payment_data.get("payment_date", datetime.now().strftime("%Y-%m-%d"))
        
        return {
            "payment_id": f"PAY-{datetime.now().strftime('%Y%m%d')}-001",
            "status": "completed",
            "amount": amount,
            "recipient": recipient,
            "payment_method": payment_method,
            "payment_date": payment_date,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(self, report_params):
        """Generate a financial report"""
        logger.info(f"Generating {report_params.get('report_type')} report")
        
        report_type = report_params.get("report_type", "income_statement")
        period = report_params.get("period", "monthly")
        format_type = report_params.get("format", "pdf")
        
        return {
            "report_id": f"REP-{datetime.now().strftime('%Y%m%d')}-001",
            "report_type": report_type,
            "period": period,
            "format": format_type,
            "status": "generated",
            "download_url": f"https://example.com/reports/REP-{datetime.now().strftime('%Y%m%d')}-001.{format_type}",
            "generated_at": datetime.now().isoformat()
        }
        
    def handle_generic_query(self, query):
        """Handle generic financial queries"""
        logger.info(f"Handling generic query: {query}")
        
        # For demo purposes, return a structured response
        return {
            "query_type": "financial_information",
            "query": query,
            "status": "processed",
            "response_time": datetime.now().isoformat()
        }

class ResponseGenerator:
    """Generates human-friendly responses based on API results"""
    
    def __init__(self, config, openai_api_key):
        self.config = config
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=self.openai_api_key)
    
    def generate_response(self, operation, result, original_query=""):
        """Generate a natural language response based on operation and result"""
        logger.info(f"Generating response for {operation} operation")
        
        prompt = f"""
        You are an AI finance accountant assistant. Create a clear, detailed response to the user about the result of their request.
        
        Original Query: "{original_query}"
        Operation: {operation}
        Result: {json.dumps(result, indent=2)}
        
        Instructions:
        1. Generate a friendly, professional response explaining the outcome of the operation
        2. Include specific details from the result (numbers, dates, names, etc.)
        3. For general queries, provide helpful financial information and advice
        4. If there was an error, explain it clearly and suggest next steps
        5. Your response should feel personalized and tailored to their specific request
        6. Keep the response informative but concise
        
        Important: Don't just repeat the raw data - interpret it and present it in a way that's meaningful to the user.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=350
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Provide a fallback response based on operation type
            if operation == "check_balance":
                account_id = result.get('account_id', 'your account')
                balance = result.get('balance', 0)
                currency = result.get('currency', 'USD')
                available = result.get('available', 0)
                return f"I checked the balance for {account_id}. The current balance is {balance} {currency}, with {available} {currency} available for use."
                
            elif operation == "create_invoice":
                client = result.get('client_name', 'the client')
                amount = result.get('amount', 0)
                invoice_id = result.get('invoice_id', '')
                due_date = result.get('due_date', 'the specified date')
                return f"I've created invoice {invoice_id} for {client} in the amount of {amount}. The invoice is due on {due_date}."
                
            elif operation == "process_payment":
                recipient = result.get('recipient', 'the recipient')
                amount = result.get('amount', 0)
                payment_id = result.get('payment_id', '')
                return f"I've processed payment {payment_id} to {recipient} for {amount}. The payment has been completed successfully."
                
            elif operation == "generate_report":
                report_type = result.get('report_type', 'financial')
                period = result.get('period', '')
                download_url = result.get('download_url', '')
                return f"I've generated your {report_type} report for the {period} period. You can download it at {download_url}."
                
            else:
                return f"I've processed your request and found the information you need. Is there anything specific you'd like to know about your finances?"

# Initialize Flask app
app = Flask(__name__)

# Initialize components
knowledge_base = KnowledgeBase(CONFIG["rag"], OPENAI_API_KEY)
command_processor = CommandProcessor(CONFIG["model"], knowledge_base, OPENAI_API_KEY)
api_connector = APIConnector(CONFIG["api"], FINANCIAL_API_KEY)
response_generator = ResponseGenerator(CONFIG["model"], OPENAI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-voice', methods=['POST'])
def process_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    audio_file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the audio file
        transcription = command_processor.process_audio(temp_file.name)
        if not transcription:
            return jsonify({'error': 'Failed to transcribe audio'}), 500
        
        # Interpret the command
        interpretation = command_processor.interpret_command(transcription)
        
        # Execute the operation
        result = api_connector.execute_operation(
            interpretation["operation"], 
            interpretation["parameters"]
        )
        
        # Generate response
        response_text = response_generator.generate_response(
            interpretation["operation"], 
            result,
            transcription
        )
        
        return jsonify({
            'transcription': transcription,
            'operation': interpretation["operation"],
            'parameters': interpretation["parameters"],
            'response': response_text
        })
        
    except Exception as e:
        logger.error(f"Error processing voice command: {str(e)}")
        return jsonify({
            'error': str(e),
            'operation': 'other',
            'response': f"I'm sorry, I encountered an error while processing your voice command. Please try again or use text input instead."
        }), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.route('/process-text', methods=['POST'])
def process_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    input_text = data['text']
    
    try:
        # Log the incoming request
        logger.info(f"Received text request: {input_text}")
        
        # Interpret the command
        interpretation = command_processor.interpret_command(input_text)
        logger.info(f"Interpretation result: {interpretation}")
        
        # Execute the operation
        result = api_connector.execute_operation(
            interpretation["operation"], 
            interpretation["parameters"]
        )
        logger.info(f"Operation result: {result}")
        
        # Generate response
        response_text = response_generator.generate_response(
            interpretation["operation"], 
            result,
            input_text
        )
        logger.info(f"Generated response: {response_text}")
        
        return jsonify({
            'operation': interpretation["operation"],
            'parameters': interpretation["parameters"],
            'response': response_text
        })
        
    except Exception as e:
        logger.error(f"Error processing text command: {str(e)}")
        return jsonify({
            'error': str(e),
            'operation': 'other',
            'response': "I'm sorry, I encountered an error while processing your request. Please try rephrasing your question."
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)