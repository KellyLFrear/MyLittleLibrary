## Step 1: Download Wikipedia Articles, Process The Text To Clean It And Remove Unwanted Articles, And Load Vocabulary Lists For Different Proficiency Levels 
from scripts.download_wiki import download_wiki
from scripts.preprocess_wiki import preprocess_wiki
from scripts.analyze_articles import analyze_articles

# Download Wikipedia Articles
download_wiki()

# Clean and Process The Wikapedia Articles
preprocess_wiki()

# Analyze The Cleaned Articles With Different Vocabulary Lists
analyze_articles()