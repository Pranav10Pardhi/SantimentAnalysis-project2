import nltk
import ssl
import os

def setup_nltk():
    # Disable SSL verification for downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    # Download required NLTK resources
    resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

if __name__ == "__main__":
    setup_nltk()