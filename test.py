#!/usr/bin/env python3
"""
Local testing utilities for Slack AI Bot
Run this script to test your knowledge base without Slack
"""

import requests
import json

BASE_URL = "http://localhost:3000"


def test_health():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("✅ Server Health Check:")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False


def add_sample_document():
    """Add a sample document to test the knowledge base"""
    doc_data = {
        "id": "test_doc_" + str(int(time.time())),
        "text": "This is a test document. Our company provides excellent customer service and technical support. We are available 24/7 to help with any issues.",
        "metadata": {"category": "test", "source": "manual_test"},
    }

    try:
        response = requests.post(f"{BASE_URL}/add_knowledge", json=doc_data)
        if response.status_code == 200:
            print("✅ Document added successfully:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to add document: {response.text}")
    except Exception as e:
        print(f"❌ Error adding document: {e}")


def list_all_documents():
    """List all documents in the knowledge base"""
    try:
        response = requests.get(f"{BASE_URL}/list_knowledge")
        print("📋 Knowledge Base Documents:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Error listing documents: {e}")


def test_search(query):
    """Test search functionality"""
    try:
        response = requests.post(f"{BASE_URL}/test_search", json={"query": query})
        print(f"🔍 Search Results for '{query}':")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Error testing search: {e}")


def interactive_test():
    """Interactive testing mode"""
    print("\n" + "=" * 50)
    print("🧪 Interactive Testing Mode")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("1. Test server health")
        print("2. Add sample document")
        print("3. List all documents")
        print("4. Test search")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            test_health()
        elif choice == "2":
            add_sample_document()
        elif choice == "3":
            list_all_documents()
        elif choice == "4":
            query = input("Enter search query: ").strip()
            if query:
                test_search(query)
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


def bulk_add_from_file(filename):
    """Add documents from a text file (one document per line)"""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if line:  # Skip empty lines
                doc_data = {
                    "id": f"bulk_doc_{i}",
                    "text": line,
                    "metadata": {
                        "source": "bulk_upload",
                        "file": filename,
                        "line_number": i + 1,
                    },
                }

                response = requests.post(f"{BASE_URL}/add_knowledge", json=doc_data)
                if response.status_code == 200:
                    print(f"✅ Added document {i+1}: {line[:50]}...")
                else:
                    print(f"❌ Failed to add document {i+1}")

        print(
            f"\n📊 Bulk upload completed: {len([l for l in lines if l.strip()])} documents processed"
        )

    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
    except Exception as e:
        print(f"❌ Error during bulk upload: {e}")


if __name__ == "__main__":
    import time
    import sys

    print("🤖 Slack AI Bot - Local Testing Utility")

    # Check if server is running
    if not test_health():
        print("\n❌ Server is not running!")
        print("Please start the bot with: python app.py")
        sys.exit(1)

    # Run interactive testing
    interactive_test()
