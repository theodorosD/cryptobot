"""
AI-Powered Crypto Trading Bot for Hackathon Demo

This script implements a continuous BTC trading bot that consults Qwen-3 LLM (via Cerebras API)
for decision-making every 30 seconds. It uses real-time BTC/EUR prices from Coinmotion API.
"""

import os
import time
import json
import requests
from cerebras.cloud.sdk import Cerebras
from colorama import Fore, init

# Initialize colorama for colored terminal output
init()

# === Configuration ===
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
COINMOTION_API_URL = "https://api.coinmotion.com/rates"
INTERVAL_SECONDS = 30

# === Initial Balances ===
EUR_BALANCE = 1200.0  # Starting EUR amount
BTC_BALANCE = 0.3     # Starting BTC amount

# === Trading Parameters ===
FEE_PERCENTAGE = 0.02  # 2% trading fee

# === Price History Tracking ===
price_history = []

# === LLM Prompts ===
SYSTEM_PROMPT = """You are an expert crypto trading advisor. 
You must ALWAYS reply with a single, valid JSON object—never include any text, comments, formatting, markdown, or greetings outside the JSON.
The JSON MUST have two keys:
- 'decision': one of 'Buy', 'Sell', or 'Hold' (case-sensitive, no other words)
- 'reasoning': a very brief explanation (20 words max) of why you chose that action, based on the latest price window and balances.
Example output:
{"decision": "Hold", "reasoning": "No clear trend and low volatility. Waiting for a better opportunity to buy or sell."}
IMPORTANT: Output ONLY the JSON object. DO NOT add explanations, headers, or code blocks. If you fail, your advice will be ignored."""

def get_current_price():
    """Fetch current BTC/EUR price from Coinmotion API"""
    try:
        response = requests.get(COINMOTION_API_URL)
        response.raise_for_status()
        data = response.json()
        current_price = float(data['btc_eur']['sell'])  # Using sell price for trading decisions
        return current_price
    except Exception as e:
        print(f"{Fore.YELLOW}Error fetching price: {str(e)}{Fore.RESET}")
        return price_history[-1] if price_history else 0  # Use last known price if API fails

def call_llm(current_price, eur_balance, btc_balance, last_buy_price):
    """Get trading advice from Qwen-3 LLM via Cerebras API"""
    try:
        client = Cerebras(api_key=CEREBRAS_API_KEY)
        
        user_prompt = f"""Given these BTC/EUR prices: {json.dumps(price_history)}, 
        current EUR balance: {eur_balance}, 
        BTC balance: {btc_balance}, 
        last buy price: {last_buy_price or 0}, 
        trading volume: {current_price}, 
        what should I do? Reply only in JSON as explained. JSON /nothink"""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="qwen-3-235b-a22b",
            stream=False,
            max_completion_tokens=256,
            temperature=1.0,
            top_p=1.0,
            response_format={"type": "json_object"}
        )
        
        # Try to parse response as JSON, handle potential parsing errors
        try:
            llm_response = json.loads(response.choices[0].message.content)
            return llm_response['decision'], llm_response['reasoning']
        except json.JSONDecodeError:
            print(f"{Fore.RED}LLM Response Not Valid JSON: {response.choices[0].message.content}{Fore.RESET}")
            return "Hold", "Invalid LLM response format"
    
    except Exception as e:
        print(f"{Fore.RED}LLM Error: {str(e)}{Fore.RESET}")
        return "Hold", "LLM error - holding by default"

def log_status(current_price, eur_balance, btc_balance, decision, reasoning):
    """Log current status with colored output"""
    color = {
        "Buy": Fore.GREEN,
        "Sell": Fore.RED,
        "Hold": Fore.YELLOW
    }.get(decision, Fore.WHITE)
    
    print(f"{Fore.CYAN}{'='*50}{Fore.RESET}")
    print(f"{Fore.BLUE}Time:{Fore.RESET} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Fore.BLUE}Current Price:{Fore.RESET} {Fore.WHITE}{current_price:.2f} €{Fore.RESET}")
    print(f"{Fore.BLUE}Balances:{Fore.RESET} {eur_balance:.2f} € / {btc_balance:.6f} BTC")
    print(f"{Fore.BLUE}Decision:{Fore.RESET} {color}{decision}{Fore.RESET}")
    print(f"{Fore.BLUE}Reasoning:{Fore.RESET} {reasoning}")
    print(f"{Fore.BLUE}Price History Length:{Fore.RESET} {len(price_history)}")
    print(f"{Fore.CYAN}{'='*50}{Fore.RESET}\n")

def execute_trade(current_price, eur_balance, btc_balance, decision):
    """Execute trade based on LLM decision"""
    global last_buy_price
    
    if decision == "Buy" and eur_balance > current_price * (1 + FEE_PERCENTAGE):
        # Calculate maximum BTC we can buy with EUR (including fee)
        eur_to_spend = eur_balance * (1 - FEE_PERCENTAGE)
        btc_bought = eur_to_spend / current_price
        new_eur_balance = 0
        new_btc_balance = btc_balance + btc_bought
        last_buy_price = current_price
        return new_eur_balance, new_btc_balance, f"Bought {btc_bought:.6f} BTC at {current_price:.2f} €"
    
    elif decision == "Sell" and btc_balance > 0:
        # Sell all BTC for EUR (including fee)
        btc_to_sell = btc_balance
        eur_received = btc_to_sell * current_price * (1 - FEE_PERCENTAGE)
        new_eur_balance = eur_balance + eur_received
        new_btc_balance = 0
        return new_eur_balance, new_btc_balance, f"Sold {btc_to_sell:.6f} BTC at {current_price:.2f} €"
    
    else:  # Hold or insufficient funds
        return eur_balance, btc_balance, "No trade executed"

def main():
    """Main trading loop"""
    global EUR_BALANCE, BTC_BALANCE, price_history
    
    print(f"{Fore.GREEN}Starting Crypto Trading Bot...{Fore.RESET}")
    
    last_buy_price = None
    
    while True:
        # Get current price and update history
        current_price = get_current_price()
        price_history.append(current_price)
        
        # Get LLM decision
        decision, reasoning = call_llm(current_price, EUR_BALANCE, BTC_BALANCE, last_buy_price)
        
        # Execute trade if possible
        EUR_BALANCE, BTC_BALANCE, trade_details = execute_trade(current_price, EUR_BALANCE, BTC_BALANCE, decision)
        
        # Log status
        log_status(current_price, EUR_BALANCE, BTC_BALANCE, decision, reasoning)
        
        # Wait for next interval
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
