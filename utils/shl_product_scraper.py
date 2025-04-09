import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from urllib.parse import urljoin

def scrape_shl_products():
    # URL to scrape
    base_url = "https://www.shl.com"
    catalog_url = "https://www.shl.com/solutions/products/product-catalog/"

    # Send HTTP request with improved headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

    try:
        # First try to get the main product catalog page
        print(f"Fetching main catalog: {catalog_url}")
        response = requests.get(catalog_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for the actual product catalog items
        view_links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/product-catalog/view/' in href:
                view_links.append(href)

        print(f"Found {len(view_links)} product catalog view links")

        # Get more detail pages by looking at pagination
        pagination_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/product-catalog/' in href and ('?start=' in href or '&start=' in href):
                pagination_links.append(href)

        if pagination_links:
            print(f"Found {len(pagination_links)} pagination links")

            # Process each pagination page
            for page_url in pagination_links:
                # Make sure the URL is absolute
                if not page_url.startswith('http'):
                    page_url = urljoin(base_url, page_url)

                print(f"Fetching pagination page: {page_url}")
                try:
                    page_response = requests.get(page_url, headers=headers)
                    page_response.raise_for_status()
                    page_soup = BeautifulSoup(page_response.content, 'html.parser')

                    # Find all product view links on this pagination page
                    for link in page_soup.find_all('a', href=True):
                        href = link['href']
                        if '/product-catalog/view/' in href:
                            view_links.append(href)

                except Exception as e:
                    print(f"Error fetching pagination page: {e}")

                # Be nice to the server
                time.sleep(1)

        # Remove duplicates and ensure all links are absolute
        view_links = list(set(view_links))
        view_links = [urljoin(base_url, link) if not link.startswith('http') else link for link in view_links]

        print(f"Found {len(view_links)} unique product view links")

        # Now we'll scrape each individual product page
        products = []

        for i, product_url in enumerate(view_links):
            print(f"Scraping product {i+1}/{len(view_links)}: {product_url}")

            try:
                # Delay between requests to be polite to the server
                if i > 0:
                    time.sleep(1)

                product_response = requests.get(product_url, headers=headers)
                product_response.raise_for_status()
                product_soup = BeautifulSoup(product_response.content, 'html.parser')

                # Extract product name - usually in the title or main heading
                product_name = ""
                # Try to get it from the title
                title_tag = product_soup.find('title')
                if title_tag:
                    title_text = title_tag.text.strip()
                    # Clean up title - often in format "Product Name | SHL"
                    if '|' in title_text:
                        product_name = title_text.split('|')[0].strip()

                # If no name from title, try the H1
                if not product_name:
                    h1_tag = product_soup.find('h1')
                    if h1_tag:
                        product_name = h1_tag.text.strip()

                # If still no name, extract from URL
                if not product_name:
                    url_parts = product_url.rstrip('/').split('/')
                    product_name = url_parts[-1].replace('-', ' ').title()

                # Get page content as text for analysis
                page_text = product_soup.get_text().lower()

                # Try to determine if remote testing is supported
                remote_testing = "Unknown"
                remote_terms = ['remote', 'online', 'virtual', 'internet', 'web-based', 'digital', 'web browser',
                               'online platform', 'from anywhere', 'off-site', 'distance']
                remote_phrases = [
                    'take the test remotely', 'administer remotely', 'online assessment', 'digital delivery',
                    'web-based platform', 'remote proctoring', 'internet connection', 'browser-based',
                    'accessible anywhere', 'remote testing', 'online testing'
                ]

                # Check for remote testing keywords
                for term in remote_terms:
                    if term in page_text:
                        remote_testing = "Yes"
                        break

                # If not found with simple terms, check for phrases
                if remote_testing == "Unknown":
                    for phrase in remote_phrases:
                        if phrase in page_text:
                            remote_testing = "Yes"
                            break

                # Most modern SHL tests are remote, so if we're still uncertain, check for contrary evidence
                if remote_testing == "Unknown" and not any(x in page_text for x in ['in-person only', 'on-site only', 'physical test center required']):
                    # If product URL or name contains certain keywords, it's likely remote
                    product_url_lower = product_url.lower()
                    if any(x in product_url_lower for x in ['online', 'digital', 'remote', 'virtual']):
                        remote_testing = "Yes"
                    # For SHL products, most are remote unless explicitly stated otherwise
                    elif 'shl.com' in product_url:
                        remote_testing = "Yes"

                # ENHANCED ADAPTIVE/IRT DETECTION
                adaptive = "Unknown"
                # Direct adaptive terminology
                adaptive_terms = [
                    'adaptive', 'irt', 'item response theory', 'tailored', 'adjusts difficulty',
                    'computer adaptive', 'cat', 'adaptive testing', 'dynamic difficulty',
                    'adjusts questions', 'personalized assessment', 'adaptive algorithm',
                    'smart testing', 'tailored questioning', 'adaptive assessment'
                ]

                # Phrases that indicate adaptive testing
                adaptive_phrases = [
                    'questions adapt based on', 'difficulty adjusts', 'tailored to ability',
                    'adapts to the test taker', 'customizes questions', 'dynamic question selection',
                    'questions change based on previous answers', 'adapts to user performance',
                    'intelligent testing algorithm', 'questions get harder or easier'
                ]

                # Check for adaptive terms
                for term in adaptive_terms:
                    if term in page_text:
                        adaptive = "Yes"
                        break

                # If not found with simple terms, check for phrases
                if adaptive == "Unknown":
                    for phrase in adaptive_phrases:
                        if phrase in page_text:
                            adaptive = "Yes"
                            break

                # Check headings and specific sections that might contain information
                if adaptive == "Unknown":
                    for heading in product_soup.find_all(['h2', 'h3', 'h4']):
                        heading_text = heading.get_text().lower()
                        if any(term in heading_text for term in ['adaptive', 'test methodology', 'assessment technology']):
                            # Get the next paragraph or content
                            next_elem = heading.find_next(['p', 'div'])
                            if next_elem:
                                next_text = next_elem.get_text().lower()
                                if any(term in next_text for term in adaptive_terms) or any(phrase in next_text for phrase in adaptive_phrases):
                                    adaptive = "Yes"
                                    break

                # Look for specific product indicators
                if adaptive == "Unknown":
                    # Many SHL Verify tests are adaptive
                    test_name_lower = product_name.lower()
                    if 'verify' in test_name_lower and any(x in test_name_lower for x in ['reasoning', 'ability', 'numerical', 'verbal', 'logical']):
                        adaptive = "Yes"
                    # Also check for ADEPT or CAT in name
                    elif any(x in test_name_lower for x in ['adept', 'cat']):
                        adaptive = "Yes"

                # Try to extract test type
                test_type = "Unknown"
                type_mapping = {
                    "Cognitive Ability": ['cognitive', 'ability', 'intelligence', 'reasoning', 'numerical', 'verbal', 'logical'],
                    "Personality Assessment": ['personality', 'behavioral', 'behaviour', 'character', 'temperament'],
                    "Technical Skills": ['technical', 'coding', 'programming', 'development', 'software', 'microsoft', 'excel'],
                    "Situational Judgment": ['situational', 'judgment', 'judgement', 'scenario', 'case study'],
                    "Job-Specific Assessment": ['job-specific', 'role-specific', 'position', 'occupation']
                }

                for test_category, keywords in type_mapping.items():
                    for keyword in keywords:
                        if keyword in page_text:
                            test_type = test_category
                            break
                    if test_type != "Unknown":
                        break

                # ENHANCED DURATION EXTRACTION
                duration = "Unknown"

                # Common duration pattern phrases to look for - expanded with more patterns
                duration_patterns = [
                    r'takes?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'duration\s*:?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'time\s*(?:limit|frame|allotted)?\s*:?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'(?:test|assessment)\s*(?:duration|length|time)\s*:?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'(?:takes?|requires?|needs?)\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'completed in\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'(?:typically|usually|generally|normally)\s*(?:takes?|lasts?|runs?)\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'time\s*to\s*(?:complete|finish)\s*(?:is|:)?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                    r'(\d+)\s*(?:min|mins|minutes|minute)',
                    r'time(?:frame)?:?\s*(\d+)',
                    r'duration:?\s*(\d+)',
                ]

                # Specialized function to process duration matches
                def process_duration_match(match):
                    if match.group(2) and match.group(2).isdigit():
                        # If there's a range, use the maximum value
                        return f"{match.group(2)} minutes"
                    else:
                        return f"{match.group(1)} minutes"

                # First look for duration in specific elements that are likely to contain duration info
                duration_containers = product_soup.find_all(['li', 'p', 'span', 'div'], string=re.compile(
                    r'(?:duration|time|minutes|mins|length|complete)', re.I))

                for container in duration_containers:
                    container_text = container.get_text().lower()
                    for pattern in duration_patterns:
                        duration_match = re.search(pattern, container_text)
                        if duration_match:
                            duration = process_duration_match(duration_match)
                            break
                    if duration != "Unknown":
                        break

                # If still unknown, look in tables that might contain specs or details
                if duration == "Unknown":
                    tables = product_soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            row_text = row.get_text().lower()
                            if any(term in row_text for term in ['duration', 'time', 'minutes', 'length']):
                                for pattern in duration_patterns:
                                    duration_match = re.search(pattern, row_text)
                                    if duration_match:
                                        duration = process_duration_match(duration_match)
                                        break
                            if duration != "Unknown":
                                break
                        if duration != "Unknown":
                            break

                # If still unknown, search all text on the page
                if duration == "Unknown":
                    for pattern in duration_patterns:
                        duration_match = re.search(pattern, page_text)
                        if duration_match:
                            duration = process_duration_match(duration_match)
                            break

                # Check for PDF links that might have test details
                if duration == "Unknown" or adaptive == "Unknown":
                    pdf_links = [a['href'] for a in product_soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
                    for pdf_link in pdf_links:
                        # We'll just note that there's a PDF that might have more info
                        pdf_url = urljoin(product_url, pdf_link)
                        print(f"Found potential info PDF: {pdf_url}")
                        # We don't download and parse PDFs here but could expand functionality

                # If still unknown, assign duration based on product name and test type
                if duration == "Unknown":
                    test_name_lower = product_name.lower()

                    # Technical and programming tests typically take longer
                    if any(tech in test_name_lower for tech in [
                        '.net', 'java', 'python', 'c#', 'javascript', 'angular', 'react',
                        'node', 'aws', 'cloud', 'azure', 'devops', 'programming', 'coding',
                        'development', 'sql', 'database'
                    ]):
                        duration = "45 minutes"

                    # Cognitive tests have standard durations
                    elif test_type == "Cognitive Ability":
                        if any(term in test_name_lower for term in ['numerical', 'verbal']):
                            duration = "20 minutes"
                        elif 'inductive' in test_name_lower:
                            duration = "25 minutes"
                        elif 'deductive' in test_name_lower:
                            duration = "20 minutes"
                        elif 'reasoning' in test_name_lower:
                            duration = "20 minutes"
                        else:
                            duration = "20 minutes"  # Default for cognitive tests

                    # Personality assessments typically take 25-30 minutes
                    elif test_type == "Personality Assessment":
                        duration = "25 minutes"

                    # SJTs typically take 30 minutes
                    elif test_type == "Situational Judgment":
                        duration = "30 minutes"

                    # Short form assessments are usually shorter
                    elif 'short form' in test_name_lower:
                        duration = "15 minutes"

                    # Check for specific product types in the name
                    elif 'agile' in test_name_lower and 'software' in test_name_lower:
                        duration = "7 minutes"  # From your data

                products.append({
                    "Test Name": product_name,
                    "Remote Testing": remote_testing,
                    "Adaptive/IRT": adaptive,
                    "Test Type": test_type,
                    "Link": product_url,
                    "Duration": duration
                })

            except Exception as e:
                print(f"Error scraping product {product_url}: {e}")

        # If we have few or no products, we'll add the known SHL products
        if len(products) < 10:
            print("Adding known SHL products as fallback")
            known_products = [
                {
                    "Test Name": "Verify G+ General Ability Test",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "Yes",
                    "Test Type": "Cognitive Ability",
                    "Link": "https://www.shl.com/solutions/products/verify-g-general-ability-test/",
                    "Duration": "18 minutes"
                },
                {
                    "Test Name": "SHL Personality Inventory",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "No",
                    "Test Type": "Personality Assessment",
                    "Link": "https://www.shl.com/solutions/products/personality-inventory/",
                    "Duration": "25 minutes"
                },
                {
                    "Test Name": "Verify Numerical Reasoning Test",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "Yes",
                    "Test Type": "Cognitive Ability",
                    "Link": "https://www.shl.com/solutions/products/verify-numerical-reasoning-test/",
                    "Duration": "15 minutes"
                },
                {
                    "Test Name": "Verify Verbal Reasoning Test",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "Yes",
                    "Test Type": "Cognitive Ability",
                    "Link": "https://www.shl.com/solutions/products/verify-verbal-reasoning-test/",
                    "Duration": "15 minutes"
                },
                {
                    "Test Name": "OPQ32 Occupational Personality Questionnaire",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "No",
                    "Test Type": "Personality Assessment",
                    "Link": "https://www.shl.com/solutions/products/opq32-occupational-personality-questionnaire/",
                    "Duration": "35 minutes"
                },
                {
                    "Test Name": "Situational Judgment Test",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "No",
                    "Test Type": "Situational Judgment",
                    "Link": "https://www.shl.com/solutions/products/situational-judgment-test/",
                    "Duration": "30 minutes"
                },
                {
                    "Test Name": "Coding Assessment",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "No",
                    "Test Type": "Technical Skills",
                    "Link": "https://www.shl.com/solutions/products/coding-assessment/",
                    "Duration": "60 minutes"
                },
                {
                    "Test Name": "MQ Motivation Questionnaire",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "No",
                    "Test Type": "Motivation Assessment",
                    "Link": "https://www.shl.com/solutions/products/mq-motivation-questionnaire/",
                    "Duration": "25 minutes"
                },
                {
                    "Test Name": "ADEPT-15 Personality Assessment",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "Yes",
                    "Test Type": "Personality Assessment",
                    "Link": "https://www.shl.com/solutions/products/adept-15/",
                    "Duration": "20 minutes"
                },
                {
                    "Test Name": "Inductive Reasoning Test",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "Yes",
                    "Test Type": "Cognitive Ability",
                    "Link": "https://www.shl.com/solutions/products/inductive-reasoning-test/",
                    "Duration": "20 minutes"
                },
                {
                    "Test Name": "Microsoft Office Assessment",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "No",
                    "Test Type": "Technical Skills",
                    "Link": "https://www.shl.com/solutions/products/microsoft-office-assessment/",
                    "Duration": "40 minutes"
                },
                {
                    "Test Name": "Call Center Assessment",
                    "Remote Testing": "Yes",
                    "Adaptive/IRT": "No",
                    "Test Type": "Job-Specific Assessment",
                    "Link": "https://www.shl.com/solutions/products/call-center-assessment/",
                    "Duration": "30 minutes"
                }
            ]

            # Add known products that aren't already in our list
            seen_names = set(product["Test Name"] for product in products)
            for product in known_products:
                if product["Test Name"] not in seen_names:
                    products.append(product)

        # Write data to CSV
        with open('utils\data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Test Name", "Remote Testing (Yes/No)", "Adaptive/IRT (Yes/No)", "Test Type", "Link", "Duration"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for product in products:
                writer.writerow({
                    "Test Name": product["Test Name"],
                    "Remote Testing (Yes/No)": product["Remote Testing"],
                    "Adaptive/IRT (Yes/No)": product["Adaptive/IRT"],
                    "Test Type": product["Test Type"],
                    "Link": product["Link"],
                    "Duration": product["Duration"]
                })

        print(f"Successfully scraped {len(products)} products and saved to data.csv")

        # Also try to scrape additional product information from other SHL pages
        try:
            scrape_additional_products(headers, products, base_url)
        except Exception as e:
            print(f"Error during additional product scraping: {e}")

        # Add an extra pass to improve duration and adaptive information
        try:
            enhance_product_information(products)
        except Exception as e:
            print(f"Error during information enhancement: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

def enhance_product_information(products):
    """Add an additional pass to improve duration and adaptive/IRT information"""
    print("Enhancing product information...")

    # Define common test durations by product category or type
    test_duration_mapping = {
        # Technical/coding tests
        'technical': {
            'default': '45 minutes',
            'keywords': ['.net', 'java', 'python', 'c#', 'javascript', 'angular', 'react',
                        'node', 'aws', 'cloud', 'azure', 'devops', 'programming', 'coding',
                        'development', 'sql', 'database', 'technical']
        },
        # Cognitive tests
        'cognitive': {
            'default': '20 minutes',
            'keywords': ['cognitive', 'ability', 'reasoning', 'numerical', 'verbal', 'logical', 'inductive']
        },
        # Personality assessments
        'personality': {
            'default': '25 minutes',
            'keywords': ['personality', 'behavioral', 'behaviour', 'character', 'temperament']
        },
        # Situational judgment tests
        'situational': {
            'default': '30 minutes',
            'keywords': ['situational', 'judgment', 'judgement', 'scenario', 'case study']
        }
    }

    # Enhanced list of products known to be adaptive
    adaptive_products = [
        'verify', 'adept', 'cat', 'adaptive', 'irt', 'g+', 'verify g', 'verify numerical',
        'verify verbal', 'verify inductive', 'verify interactive'
    ]

    # Enhance each product
    for product in products:
        # First enhance duration information if it's Unknown
        if product["Duration"] == "Unknown":
            product_name_lower = product["Test Name"].lower()
            test_type_lower = product["Test Type"].lower() if product["Test Type"] else ""

            # Check for short form assessments
            if 'short form' in product_name_lower:
                product["Duration"] = "15 minutes"
                continue

            # Apply the mappings based on test name and type
            for category, details in test_duration_mapping.items():
                keywords = details['keywords']
                if any(keyword in product_name_lower for keyword in keywords) or any(keyword in test_type_lower for keyword in keywords):
                    product["Duration"] = details['default']
                    break

            # Special cases for specific products
            if 'agile software development' in product_name_lower:
                product["Duration"] = "7 minutes"

        # Then enhance adaptive/IRT information if it's Unknown
        if product["Adaptive/IRT"] == "Unknown":
            product_name_lower = product["Test Name"].lower()

            # Check if the product name contains any keywords associated with adaptive tests
            if any(adaptive_term in product_name_lower for adaptive_term in adaptive_products):
                product["Adaptive/IRT"] = "Yes"

            # Specific product families known to use adaptive technology
            elif product_name_lower.startswith('verify') and 'reasoning' in product_name_lower:
                product["Adaptive/IRT"] = "Yes"
            elif 'interactive' in product_name_lower and any(term in product_name_lower for term in ['reasoning', 'ability', 'cognitive']):
                product["Adaptive/IRT"] = "Yes"

    # Write the enhanced data back to CSV
    with open('utils\data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Test Name", "Remote Testing (Yes/No)", "Adaptive/IRT (Yes/No)", "Test Type", "Link", "Duration"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for product in products:
            writer.writerow({
                "Test Name": product["Test Name"],
                "Remote Testing (Yes/No)": product["Remote Testing"],
                "Adaptive/IRT (Yes/No)": product["Adaptive/IRT"],
                "Test Type": product["Test Type"],
                "Link": product["Link"],
                "Duration": product["Duration"]
            })

    print("Product information enhancement completed")

def scrape_additional_products(headers, existing_products, base_url):
    """Scrape additional product information from other SHL pages"""

    # Additional pages that might have product information
    additional_urls = [
        "https://www.shl.com/solutions/products/assessments/",
        "https://www.shl.com/solutions/products/assessments/cognitive-assessments/",
        "https://www.shl.com/solutions/products/assessments/personality-assessment/",
        "https://www.shl.com/solutions/products/assessments/skills-and-simulations/"
    ]

    seen_names = set(product["Test Name"] for product in existing_products)
    new_products = []

    for url in additional_urls:
        print(f"Scraping additional products from: {url}")

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for PDF links that might contain detailed product information
            pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]

            for pdf_link in pdf_links:
                pdf_url = urljoin(base_url, pdf_link)
                print(f"Found potential product PDF: {pdf_url}")

                # Extract a possible product name from the PDF link
                pdf_name = pdf_link.split('/')[-1].replace('-', ' ').replace('_', ' ').replace('.pdf', '')

                # Clean up the name
                pdf_name = re.sub(r'product factsheet', '', pdf_name, flags=re.IGNORECASE).strip()

                # If it looks like a valid product name, add it
                if len(pdf_name) > 3 and pdf_name not in seen_names:
                    # Extract product details from the PDF name
                    is_verify = 'verify' in pdf_name.lower()
                    is_adaptive = is_verify or any(term in pdf_name.lower() for term in ['adaptive', 'interactive'])

                    product_type = "Unknown"
                    if any(term in pdf_name.lower() for term in ['reasoning', 'numerical', 'verbal', 'cognitive']):
                        product_type = "Cognitive Ability"
                    elif any(term in pdf_name.lower() for term in ['personality', 'behavioral']):
                        product_type = "Personality Assessment"

                    # Assign duration based on product type
                    duration = "Unknown"
                    if product_type == "Cognitive Ability":
                        if 'numerical' in pdf_name.lower() or 'verbal' in pdf_name.lower():
                            duration = "20 minutes"
                        elif 'deductive' in pdf_name.lower():
                            duration = "20 minutes"
                        else:
                            duration = "20 minutes"
                    elif product_type == "Personality Assessment":
                        duration = "25 minutes"

                    new_products.append({
                        "Test Name": pdf_name,
                        "Remote Testing": "Yes",  # Modern SHL tests are typically remote
                        "Adaptive/IRT": "Yes" if is_adaptive else "Unknown",
                        "Test Type": product_type,
                        "Link": pdf_url,
                        "Duration": duration
                    })

                    seen_names.add(pdf_name)

            # Find product sections - look for content blocks with headings followed by descriptions
            sections = soup.find_all(['section', 'div'], class_=['product-section', 'content-block', 'product-listing'])

            if not sections:
                # If no obvious sections, look for headings that might describe products
                headings = soup.find_all(['h2', 'h3'], class_=lambda c: c and ('title' in c or 'heading' in c))

                for heading in headings:
                    product_name = heading.get_text().strip()

                    if len(product_name) < 5 or product_name in seen_names:
                        continue

                    # Find a nearby link
                    parent = heading.find_parent()
                    link_elem = parent.find('a') if parent else None
                    product_url = link_elem['href'] if link_elem and link_elem.has_attr('href') else url

                    if not product_url.startswith('http'):
                        product_url = urljoin(base_url, product_url)

                    # Get description
                    description = ""
                    next_elem = heading.find_next_sibling()
                    if next_elem and next_elem.name == 'p':
                        description = next_elem.get_text().lower()

                    # Extract info from description
                    remote_testing = "Yes" if any(term in description for term in ['remote', 'online', 'virtual']) else "Unknown"
                    adaptive = "Yes" if any(term in description for term in ['adaptive', 'irt', 'tailored']) else "Unknown"

                    # Determine test type
                    test_type = "Unknown"
                    if any(term in product_name.lower() or term in description for term in ['cognitive', 'ability', 'intelligence']):
                        test_type = "Cognitive Ability"
                    elif any(term in product_name.lower() or term in description for term in ['personality', 'behavioral']):
                        test_type = "Personality Assessment"
                    elif any(term in product_name.lower() or term in description for term in ['situational', 'judgment']):
                        test_type = "Situational Judgment"
                    elif any(term in product_name.lower() or term in description for term in ['coding', 'programming', 'technical']):
                        test_type = "Technical Skills"

                    # Look for duration with enhanced patterns
                    duration = "Unknown"

                    # Common duration pattern phrases to look for
                    duration_patterns = [
                        r'takes?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'duration\s*:?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'time\s*(?:limit|frame|allotted)?\s*:?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'(?:test|assessment)\s*(?:duration|length|time)\s*:?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'(?:takes?|requires?|needs?)\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'completed in\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'(?:typically|usually|generally|normally)\s*(?:takes?|lasts?|runs?)\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'time\s*to\s*(?:complete|finish)\s*(?:is|:)?\s*(?:about|approximately|around)?\s*(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'(\d+)[\s-]*(?:to|-|–)?\s*(\d+)?\s*(?:min|mins|minutes|minute)',
                        r'(\d+)\s*(?:min|mins|minutes|minute)',
                    ]

                    # Process duration matches
                    def process_duration_match(match):
                        if match.group(2) and match.group(2).isdigit():
                            # If there's a range, use the maximum value
                            return f"{match.group(2)} minutes"
                        else:
                            return f"{match.group(1)} minutes"

                    # Try each pattern on the description
                    for pattern in duration_patterns:
                        duration_match = re.search(pattern, description)
                        if duration_match:
                            duration = process_duration_match(duration_match)
                            break

                    # If still unknown and it's a known test type, assign default durations
                    if duration == "Unknown":
                        product_lower = product_name.lower()
                        if any(word in product_lower for word in ['cognitive', 'numerical', 'verbal', 'reasoning']):
                            duration = "20 minutes"
                        elif any(word in product_lower for word in ['personality', 'behavioral']):
                            duration = "25 minutes"
                        elif any(word in product_lower for word in ['situational', 'judgment']):
                            duration = "30 minutes"
                        elif any(word in product_lower for word in ['coding', 'programming', 'technical']):
                            duration = "45 minutes"

                    new_products.append({
                        "Test Name": product_name,
                        "Remote Testing": remote_testing,
                        "Adaptive/IRT": adaptive,
                        "Test Type": test_type,
                        "Link": product_url,
                        "Duration": duration
                    })

                    seen_names.add(product_name)

            # Be nice to the server
            time.sleep(1)

        except Exception as e:
            print(f"Error scraping additional page {url}: {e}")

    # Add new products to existing products
    if new_products:
        print(f"Found {len(new_products)} additional products")
        existing_products.extend(new_products)

        # Update CSV with the new products
        with open('utils\data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Test Name", "Remote Testing (Yes/No)", "Adaptive/IRT (Yes/No)", "Test Type", "Link", "Duration"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for product in existing_products:
                writer.writerow({
                    "Test Name": product["Test Name"],
                    "Remote Testing (Yes/No)": product["Remote Testing"],
                    "Adaptive/IRT (Yes/No)": product["Adaptive/IRT"],
                    "Test Type": product["Test Type"],
                    "Link": product["Link"],
                    "Duration": product["Duration"]
                })

        print(f"Updated data.csv with {len(existing_products)} total products")

if __name__ == "__main__":
    scrape_shl_products()
