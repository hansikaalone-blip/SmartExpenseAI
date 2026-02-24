import os
import re
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

creds = None

# Authentication
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
else:
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

service = build('gmail', 'v1', credentials=creds)

# Fetch recent transaction emails (last 30 days)
results = service.users().messages().list(
    userId='me',
    q="(debited OR spent OR INR OR Rs) newer_than:30d",
    maxResults=20
).execute()

messages = results.get('messages', [])

print(f"\nFound {len(messages)} possible transaction emails\n")

transactions = []

for msg in messages:
    txt = service.users().messages().get(
        userId='me',
        id=msg['id'],
        format='full'
    ).execute()

    data = ""
    if 'parts' in txt['payload']:
        for part in txt['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                data = base64.urlsafe_b64decode(
                    part['body']['data']).decode('utf-8')
    else:
        data = base64.urlsafe_b64decode(
            txt['payload']['body']['data']).decode('utf-8')

    match = re.search(r'(Rs\.?|INR)\s?(\d+)', data)

    if match:
        amount = int(match.group(2))

        merchant_match = re.search(r'at\s([A-Za-z]+)|to\s([A-Za-z]+)', data)
        if merchant_match:
            merchant = merchant_match.group(1) if merchant_match.group(1) else merchant_match.group(2)
        else:
            merchant = "Unknown"

        transactions.append({"amount": amount, "merchant": merchant})

# Categorization
def categorize(merchant):
    merchant = merchant.lower()
    if merchant in ["zomato", "swiggy"]:
        return "Food"
    elif merchant in ["amazon", "flipkart"]:
        return "Shopping"
    elif merchant in ["uber", "ola"]:
        return "Travel"
    else:
        return "Others"

for t in transactions:
    t["category"] = categorize(t["merchant"])

monthly_budget = 10000
total_spent = sum(t["amount"] for t in transactions)

print("\n===== SMART EXPENSE DASHBOARD =====\n")

for t in transactions:
    print(f"{t['merchant']:10} | Rs.{t['amount']:6} | {t['category']}")

print("\nTotal Spent:", total_spent)
print("Budget:", monthly_budget)

if total_spent > monthly_budget:
    print("⚠ Budget Exceeded!")
else:
    print("✅ Within Budget")

# ML Prediction
if len(transactions) > 1:
    days = np.array(range(1, len(transactions)+1)).reshape(-1,1)
    amounts = np.array([t["amount"] for t in transactions])
    model = LinearRegression()
    model.fit(days, amounts)
    predicted = model.predict([[len(transactions)+1]])[0]
    print("\nPredicted Next Expense:", round(predicted,2))

# Visualization
category_totals = {}
for t in transactions:
    category_totals[t["category"]] = category_totals.get(t["category"], 0) + t["amount"]

plt.bar(category_totals.keys(), category_totals.values())
plt.title("Spending by Category")
plt.xlabel("Category")
plt.ylabel("Rs.Amount")
plt.show()