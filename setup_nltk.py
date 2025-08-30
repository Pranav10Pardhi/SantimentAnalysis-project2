import nltk
import ssl

def download_nltk_data():
    try:
        # Handle SSL certificate verification
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass

        # Download required NLTK data
        packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for package in packages:
            nltk.download(package, quiet=True)
        print("NLTK data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    download_nltk_data()