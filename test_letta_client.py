#!/usr/bin/env python3
"""
Test script to isolate the Letta client URL issue
"""
import os
import sys

# Set up environment before importing anything
os.environ['LETTA_BASE_URL'] = 'http://localhost:8283'

print("=== Letta Client URL Test ===")
print(f"Initial LETTA_BASE_URL: {os.getenv('LETTA_BASE_URL')}")

try:
    # Import the letta client
    from letta_client import AsyncLetta

    print(f"After import LETTA_BASE_URL: {os.getenv('LETTA_BASE_URL')}")

    # Create client with explicit HTTP URL
    print("Creating AsyncLetta client with base_url='http://localhost:8283'")
    client = AsyncLetta(base_url='http://localhost:8283')

    print(f"After client creation LETTA_BASE_URL: {os.getenv('LETTA_BASE_URL')}")

    # Check if client has a base_url attribute
    if hasattr(client, 'base_url'):
        print(f"Client base_url: {client.base_url}")
    else:
        print("Client has no base_url attribute")

    # Check client configuration
    print(f"Client config: {client.__dict__}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()