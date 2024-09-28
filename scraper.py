import streamlit as st
import os
import json
import itertools
from pathlib import Path
from paperscraper import dump_queries
from paperscraper.pdf import save_pdf
import time
import threading

# Function to read all JSON files from a directory and return the papers as a list
def load_papers_from_directory(directory):
    papers = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):  # Assuming papers are saved in JSONL format
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        papers.append(json.loads(line))
    return papers

# Function to run the dump_queries in a separate thread
def run_scraper(keywords, directory):
    queries = [list(itertools.permutations(keywords, r)) for r in range(1, len(keywords) + 1)]
    flat_queries = [list(query) for sublist in queries for query in sublist]
    dump_queries(flat_queries, directory)

# Function to download the PDF of a paper
def download_paper_pdf(doi, title):
    paper_data = {'doi': doi}
    filename = f"{title.replace(' ', '_').replace('/', '_')}.pdf"  # Save file as sanitized title.pdf
    save_pdf(paper_data, filepath=filename)
    return filename

# Streamlit UI and functionality
def run():
    st.title("SAGE")
    st.header("Search Academic Papers")

    # Input field for search query (only one text input)
    search_query = st.text_input('Enter search terms', '')

    # Convert the search terms into a list of keywords
    keywords = [term.strip() for term in search_query.split(' ') if term]

    # Directory to save papers
    output_dir = 'arxiv'  # Assuming 'arxiv' directory already exists

    # Check if scraping is already running
    if 'scraping_thread' not in st.session_state:
        st.session_state.scraping_thread = None

    # Track displayed papers
    if 'displayed_papers' not in st.session_state:
        st.session_state.displayed_papers = set()

    # If the search button is clicked
    if st.button("Search and Save Papers"):
        if not keywords:
            st.warning("Please enter valid search terms.")
        else:
            # If there is no active thread running the scraping process
            if st.session_state.scraping_thread is None or not st.session_state.scraping_thread.is_alive():
                # Start scraping in a new thread
                st.session_state.scraping_thread = threading.Thread(target=run_scraper, args=(keywords, '.'))
                st.session_state.scraping_thread.start()
                st.success("Started fetching papers. They will appear below.")
            else:
                st.warning("Scraping is already in progress. Please wait for it to complete.")

    # Display papers from the saved directory dynamically
    while True:
        papers = load_papers_from_directory(output_dir)
        new_papers = [p for p in papers if p.get('doi') not in st.session_state.displayed_papers]  # Use DOI as the unique identifier
        if new_papers:
            st.write(f"Displaying {len(new_papers)} new papers:")
            for paper in new_papers:
                title = paper.get('title', 'No Title')
                doi = paper.get('doi', '#')
                if doi == '#':
                    continue  # Skip papers without DOI

                st.subheader(title)
                st.write(f"**Authors:** {''.join(paper.get('authors', []))}")
                st.write(f"**Published:** {paper.get('date', 'No Date')}")
                st.write(f"**Source:** {paper.get('journal', 'Unknown')}")
                st.write(f"**Link:** [View Paper]({doi.split('arXiv.')[-1]})")
                with st.expander("Show Abstract"):
                    st.write(paper.get('abstract', 'No Abstract Available'))

                # Generate a unique key for each download button
                button_key = f"download_{doi}"  # Add timestamp to ensure uniqueness
                if st.button(f"Download PDF of {title}", key=button_key):
                    pdf_filename = download_paper_pdf(doi, title)
                    st.success(f"Downloaded {pdf_filename}")
                    st.session_state.displayed_papers.add(doi)

                st.write("---")

        time.sleep(5)  # Poll every 5 seconds to check for new papers

if __name__ == "__main__":
    run()